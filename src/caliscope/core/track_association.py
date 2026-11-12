"""Link camera-local tracks into cross-camera objects by geometry.

Every track starts unlinked (negative object_id). associate_tracks uses known
extrinsics to decide which tracks in different cameras show the same object,
and returns the decision as TrackAssociations. BoxTracks.apply_associations then
writes the linked object IDs onto the boxes.

Without extrinsics, tracks cannot be linked by geometry. BoxTracks.primary_tracks
picks one track per camera by bounding box footprint instead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import rtoml
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.box_tracks import BoxTracks, _covered_area
from caliscope.core.point_data import ImagePoints, _validate_dataframe
from caliscope.exceptions import CalibrationError
from caliscope.persistence import PersistenceError, _safe_write_toml

logger = logging.getLogger(__name__)

# Largest pooled epipolar distance, as a fraction of box size, at which two tracks still count as agreeing.
# Measured on real footage with reference rigs: tracks of one person reach 0.08 between cameras facing each
# other, and every two- and three-camera subset linked correctly for values from 0.08 to 0.20.
MAX_PAIR_DISTANCE = 0.12
# Sampled moments two tracks in different cameras must share before they are tested as a seed pair.
MIN_SHARED_MOMENTS = 6
# Shared moments two tracks need before their distance counts as evidence for or against a link.
MIN_SUPPORT_MOMENTS = 3
# Shared keypoints that make one moment count.
MIN_KEYPOINTS_PER_MOMENT = 3
# Share of a track's evaluated observations that must be inliers for it to join an object.
MIN_INLIER_FRACTION = 0.5
# A competing merge with at least this share of a merge's gain makes it ambiguous.
AMBIGUITY_RATIO = 0.8
# Smallest line angle between back-projected rays for an observation to count as well conditioned. Ranking only.
MIN_RAY_ANGLE_DEG = 15.0

NO_KEYPOINTS_SAMPLED = "no keypoints sampled"
NO_EXTRINSICS = "camera has no extrinsics"
NO_GEOMETRIC_SUPPORT = "no geometric support"
CONFLICTING_TRACK = "conflicts with track {other} in the same camera"
TOO_LITTLE_SUPPORT = "too little support to join object {object_id}"

LINKED_COLUMNS: dict[str, dict[str, object]] = {
    "object_id": {"dtype": "int", "nullable": False},
    "cam_id": {"dtype": "int", "nullable": False},
    "track_id": {"dtype": "int", "nullable": False},
    "n_moments": {"dtype": "int", "nullable": True},
    "n_observations": {"dtype": "int", "nullable": True},
    "inlier_fraction": {"dtype": "float", "nullable": True},
    "median_reprojection_px": {"dtype": "float", "nullable": True},
}

UNRESOLVED_COLUMNS: dict[str, dict[str, object]] = {
    "cam_id": {"dtype": "int", "nullable": False},
    "track_id": {"dtype": "int", "nullable": False},
    "reason": {"dtype": "str", "nullable": False},
}

KEYPOINT_INLIER_RATE_COLUMNS: dict[str, dict[str, object]] = {
    "object_id": {"dtype": "int", "nullable": False},
    "keypoint_id": {"dtype": "int", "nullable": False},
    "n_observations": {"dtype": "int", "nullable": False},
    "inlier_rate": {"dtype": "float", "nullable": False},
}

_TrackKey = tuple[int, int]


def _empty_table(schema: Mapping[str, object]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(schema))


def _validated_table(
    df: pd.DataFrame, schema: dict[str, dict[str, object]], name: str, sort_by: list[str]
) -> pd.DataFrame:
    table = _validate_dataframe(df.copy(), schema, name)[list(schema)]
    return table.sort_values(sort_by, kind="stable").reset_index(drop=True)


@dataclass(frozen=True, eq=False)
class TrackAssociations:
    """Which camera-local tracks form each cross-camera object, and why the rest do not.

    linked has one row per linked track with its object_id (0 and up) and, when
    produced by associate_tracks, the geometric evidence for it. unresolved has
    one row per track that was not linked, with a reason. keypoint_inlier_rates
    has one row per object and keypoint. A (cam_id, track_id) appears at most
    once across linked and unresolved.
    """

    _linked: pd.DataFrame
    _unresolved: pd.DataFrame = field(default_factory=lambda: _empty_table(UNRESOLVED_COLUMNS))
    _keypoint_inlier_rates: pd.DataFrame = field(default_factory=lambda: _empty_table(KEYPOINT_INLIER_RATE_COLUMNS))

    def __post_init__(self) -> None:
        linked = _validated_table(
            self._linked, LINKED_COLUMNS, "TrackAssociations.linked", ["object_id", "cam_id", "track_id"]
        )
        unresolved = _validated_table(
            self._unresolved, UNRESOLVED_COLUMNS, "TrackAssociations.unresolved", ["cam_id", "track_id"]
        )
        rates = _validated_table(
            self._keypoint_inlier_rates,
            KEYPOINT_INLIER_RATE_COLUMNS,
            "TrackAssociations.keypoint_inlier_rates",
            ["object_id", "keypoint_id"],
        )

        if (linked["object_id"] < 0).any():
            raise ValueError("TrackAssociations validation failed: linked object_id values must be 0 or greater")

        keys = pd.concat([linked[["cam_id", "track_id"]], unresolved[["cam_id", "track_id"]]], ignore_index=True)
        repeated = keys[keys.duplicated()]
        if not repeated.empty:
            shown = ", ".join(f"({c}, {t})" for c, t in repeated.drop_duplicates().itertuples(index=False))
            raise ValueError(
                f"TrackAssociations validation failed: (cam_id, track_id) {shown} appear more than once "
                "across linked and unresolved tracks"
            )

        object.__setattr__(self, "_linked", linked)
        object.__setattr__(self, "_unresolved", unresolved)
        object.__setattr__(self, "_keypoint_inlier_rates", rates)

    @property
    def linked(self) -> pd.DataFrame:
        """Return a copy of the linked-track table."""
        return self._linked.copy()

    @property
    def unresolved(self) -> pd.DataFrame:
        """Return a copy of the unresolved-track table."""
        return self._unresolved.copy()

    @property
    def keypoint_inlier_rates(self) -> pd.DataFrame:
        """Return a copy of the per-object keypoint inlier-rate table."""
        return self._keypoint_inlier_rates.copy()

    @classmethod
    def from_mapping(cls, mapping: Mapping[int, Mapping[int, Iterable[int]]]) -> TrackAssociations:
        """Link tracks by hand: {object_id: {cam_id: [track_ids]}}.

        Statistics are left empty and there are no unresolved rows.

        Raises:
            ValueError: on a negative object_id or a track listed under two objects.
        """
        rows: list[dict[str, int | None]] = []
        owners: dict[_TrackKey, int] = {}
        for object_id, tracks_by_camera in sorted(mapping.items()):
            if object_id < 0:
                raise ValueError(f"object_id must be 0 or greater, got {object_id}")
            for cam_id, track_ids in sorted(tracks_by_camera.items()):
                for track_id in sorted(set(track_ids)):
                    key = (int(cam_id), int(track_id))
                    if key in owners:
                        raise ValueError(
                            f"(cam_id, track_id) {key} is listed under objects {owners[key]} and {object_id}"
                        )
                    owners[key] = int(object_id)
                    rows.append({"object_id": int(object_id), "cam_id": key[0], "track_id": key[1]})
        return cls(pd.DataFrame(rows, columns=list(LINKED_COLUMNS)))

    def to_toml(self, path: str | Path) -> None:
        """Write one TOML file atomically. An empty statistic is an omitted key.

        Raises:
            PersistenceError: if the file cannot be written.
        """
        path = Path(path)
        data: dict[str, list[dict[str, int | float | str]]] = {}
        for section, table in (
            ("linked", self._linked),
            ("unresolved", self._unresolved),
            ("keypoint_inlier_rate", self._keypoint_inlier_rates),
        ):
            records = _toml_records(table)
            if records:
                data[section] = records
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _safe_write_toml(data, path)
        except Exception as e:
            raise PersistenceError(f"Failed to save track associations to {path}: {e}") from e

    @classmethod
    def from_toml(cls, path: str | Path) -> TrackAssociations:
        """Read a file written by to_toml.

        Raises:
            PersistenceError: if the file is absent or malformed.
        """
        path = Path(path)
        if not path.exists():
            raise PersistenceError(f"Track associations file not found: {path}")
        try:
            data = rtoml.load(path)
            return cls(
                pd.DataFrame(data.get("linked", []), columns=list(LINKED_COLUMNS)),
                pd.DataFrame(data.get("unresolved", []), columns=list(UNRESOLVED_COLUMNS)),
                pd.DataFrame(data.get("keypoint_inlier_rate", []), columns=list(KEYPOINT_INLIER_RATE_COLUMNS)),
            )
        except Exception as e:
            raise PersistenceError(f"Failed to load track associations from {path}: {e}") from e


def _toml_records(table: pd.DataFrame) -> list[dict[str, int | float | str]]:
    """Rows as plain Python values, leaving out null cells because TOML has no null."""
    records: list[dict[str, int | float | str]] = []
    for row in table.to_dict(orient="records"):
        record: dict[str, int | float | str] = {}
        for column, value in row.items():
            if pd.isna(value):
                continue
            if isinstance(value, (int, np.integer)):
                record[str(column)] = int(value)
            elif isinstance(value, (float, np.floating)):
                record[str(column)] = float(value)
            else:
                record[str(column)] = str(value)
        records.append(record)
    return records


def associate_tracks(
    unlinked_points: ImagePoints,
    cameras: CameraArray,
    boxes: BoxTracks,
    *,
    max_reprojection_px: float = 8.0,
) -> TrackAssociations:
    """Link camera-local tracks into cross-camera objects with fixed extrinsics.

    Every track in boxes appears once in the result, either linked or
    unresolved with a reason. Tracks in cameras without extrinsics are
    unresolved with "camera has no extrinsics". A track belongs to at most one
    object, and the tracks of one object never overlap in time within a camera.
    Objects are numbered by box footprint summed across cameras, largest
    first, using the full-rate boxes. Camera parameters are never changed.

    Two tracks agree when their keypoints lie near each other's epipolar lines,
    averaged over the sampled moments they share and measured as a fraction of
    box size (MAX_PAIR_DISTANCE). A track joins an object on its agreement with
    all the object's tracks together, so one poor camera pair does not split an
    object. When two overlapping tracks in a camera fit an object about equally
    well, both stay unresolved.

    max_reprojection_px does not decide which tracks link. It sets which
    observations count as inliers in the reported n_moments, n_observations,
    inlier_fraction and keypoint_inlier_rates.

    Raises:
        ValueError: if track_id is missing or null, if any row is already
            linked (object_id >= 0), if points name a (cam_id, track_id)
            absent from boxes, or if max_reprojection_px is not finite and > 0.
        CalibrationError: if a camera in the points lacks intrinsics, or if
            fewer than two cameras in the points have extrinsics. The message
            points at bootstrap_extrinsics.
    """
    _validate_association_inputs(unlinked_points, cameras, boxes, max_reprojection_px)

    box_df = boxes.df
    track_moments = _moments_by_track(box_df)

    result = _link_tracks(unlinked_points, cameras, box_df, track_moments, max_reprojection_px)
    object_ids = _rank_by_footprint(result.objects, box_df)

    tracks_with_points = set(_moments_by_track(unlinked_points.df))
    posed = cameras.posed_cameras

    linked_rows: list[dict[str, int | float]] = []
    rate_rows: list[dict[str, int | float]] = []
    linked_keys: set[_TrackKey] = set()
    for index, members in enumerate(result.objects):
        keypoint_totals: dict[int, list[int]] = {}
        for key in sorted(members):
            support = result.member_support[key]
            linked_keys.add(key)
            linked_rows.append(
                {
                    "object_id": object_ids[index],
                    "cam_id": key[0],
                    "track_id": key[1],
                    "n_moments": support.n_moments,
                    "n_observations": support.n_observations,
                    "inlier_fraction": support.inlier_fraction,
                    "median_reprojection_px": support.median_px,
                }
            )
            for keypoint_id, (n_evaluated, n_inliers) in support.keypoint_counts.items():
                totals = keypoint_totals.setdefault(keypoint_id, [0, 0])
                totals[0] += n_evaluated
                totals[1] += n_inliers
        for keypoint_id, (n_evaluated, n_inliers) in sorted(keypoint_totals.items()):
            rate_rows.append(
                {
                    "object_id": object_ids[index],
                    "keypoint_id": keypoint_id,
                    "n_observations": n_evaluated,
                    "inlier_rate": n_inliers / n_evaluated,
                }
            )

    unresolved_rows: list[dict[str, int | str]] = []
    for key in sorted(track_moments):
        if key in linked_keys:
            continue
        if key[0] not in posed:
            reason = NO_EXTRINSICS
        elif key not in tracks_with_points:
            reason = NO_KEYPOINTS_SAMPLED
        else:
            reason = result.unresolved.get(key, _Unresolved(NO_GEOMETRIC_SUPPORT)).describe(object_ids)
        unresolved_rows.append({"cam_id": key[0], "track_id": key[1], "reason": reason})

    logger.info(
        f"associate_tracks: linked {len(linked_rows)} tracks into {len(result.objects)} objects; "
        f"{len(unresolved_rows)} tracks unresolved"
    )
    return TrackAssociations(
        pd.DataFrame(linked_rows, columns=list(LINKED_COLUMNS)),
        pd.DataFrame(unresolved_rows, columns=list(UNRESOLVED_COLUMNS)),
        pd.DataFrame(rate_rows, columns=list(KEYPOINT_INLIER_RATE_COLUMNS)),
    )


def _validate_association_inputs(
    points: ImagePoints, cameras: CameraArray, boxes: BoxTracks, max_reprojection_px: float
) -> None:
    if not np.isfinite(max_reprojection_px) or max_reprojection_px <= 0:
        raise ValueError(f"max_reprojection_px must be finite and > 0, got {max_reprojection_px}")

    df = points.df
    if "track_id" not in df.columns:
        raise ValueError(
            "associate_tracks needs a track_id on every point; estimate keypoints from tracked boxes first"
        )
    n_linked = int((df["object_id"] >= 0).sum())
    if n_linked:
        raise ValueError(
            f"associate_tracks takes unlinked points only, but {n_linked} rows already have a linked "
            "object_id (0 or greater)"
        )

    absent = sorted(set(_moments_by_track(df)) - set(_moments_by_track(boxes.df)))
    if absent:
        raise ValueError(f"points name (cam_id, track_id) {absent} absent from the boxes")

    cam_ids = sorted(int(c) for c in df["cam_id"].unique().tolist())
    missing = [c for c in cam_ids if c not in cameras.cameras]
    if missing:
        raise CalibrationError(f"points name cameras {missing} absent from the camera array")
    without_intrinsics = [
        c for c in cam_ids if cameras.cameras[c].matrix is None or cameras.cameras[c].distortions is None
    ]
    if without_intrinsics:
        raise CalibrationError(f"cameras {without_intrinsics} lack intrinsics; calibrate intrinsics first")

    posed = [c for c in cam_ids if c in cameras.posed_cameras]
    if len(posed) < 2:
        raise CalibrationError(
            f"associate_tracks needs extrinsics for at least two cameras in the points, found {posed}. "
            "Estimate initial extrinsics with bootstrap_extrinsics first."
        )


def _moments_by_track(df: pd.DataFrame) -> dict[_TrackKey, frozenset[int]]:
    """The sync moments at which each (cam_id, track_id) has a row."""
    moments: dict[_TrackKey, set[int]] = {}
    columns = zip(df["cam_id"].tolist(), df["track_id"].tolist(), df["sync_index"].tolist())
    for cam_id, track_id, sync_index in columns:
        moments.setdefault((int(cam_id), int(track_id)), set()).add(int(sync_index))
    return {key: frozenset(moments[key]) for key in sorted(moments)}


def _rank_by_footprint(objects: Sequence[frozenset[_TrackKey]], box_df: pd.DataFrame) -> list[int]:
    """Number objects by summed per-camera box union, largest first; ties go to the smallest member."""
    ranking: list[tuple[int, _TrackKey, int]] = []
    for index, members in enumerate(objects):
        total = 0
        for cam_id in sorted({cam_id for cam_id, _ in members}):
            track_ids = [track_id for c, track_id in members if c == cam_id]
            rows = box_df[(box_df["cam_id"] == cam_id) & box_df["track_id"].isin(track_ids)]
            total += _covered_area(rows)
        ranking.append((-total, min(members), index))

    object_ids = [0] * len(objects)
    for object_id, (_, _, index) in enumerate(sorted(ranking)):
        object_ids[index] = object_id
    return object_ids


###############################################################################
# Linking core
###############################################################################


@dataclass(frozen=True)
class _Support:
    """How well one track agrees with a set of member tracks in other cameras."""

    n_moments: int
    n_conditioned_moments: int  # moments whose inliers also cross at MIN_RAY_ANGLE_DEG or more
    n_observations: int
    n_evaluated: int
    median_px: float
    keypoint_counts: dict[int, tuple[int, int]]  # keypoint_id -> (evaluated, inliers)

    @property
    def inlier_fraction(self) -> float:
        return self.n_observations / self.n_evaluated if self.n_evaluated else 0.0

    @property
    def passes(self) -> bool:
        return self.n_moments >= MIN_SUPPORT_MOMENTS and self.inlier_fraction >= MIN_INLIER_FRACTION

    @property
    def rank(self) -> tuple[int, int, int, float]:
        """Sort key, strongest first: more well-conditioned moments, more moments, more inliers, lower residual."""
        return (-self.n_conditioned_moments, -self.n_moments, -self.n_observations, self.median_px)


_NO_SUPPORT = _Support(
    n_moments=0, n_conditioned_moments=0, n_observations=0, n_evaluated=0, median_px=float("inf"), keypoint_counts={}
)


@dataclass(frozen=True)
class _Unresolved:
    reason: str
    other_track_id: int | None = None
    object_index: int | None = None

    def describe(self, object_ids: Sequence[int]) -> str:
        object_id = object_ids[self.object_index] if self.object_index is not None else None
        return self.reason.format(other=self.other_track_id, object_id=object_id)


@dataclass(frozen=True)
class _LinkResult:
    """Objects as sets of (cam_id, track_id), in discovery order, with the evidence for each member."""

    objects: list[frozenset[_TrackKey]]
    member_support: dict[_TrackKey, _Support]
    unresolved: dict[_TrackKey, _Unresolved]


@dataclass(frozen=True)
class _Rig:
    """Normalized projection matrices and focal lengths of the cameras with extrinsics."""

    cam_ids: tuple[int, ...]
    projections: NDArray[np.float64]  # (n_cameras, 3, 4), [R | t]
    focal_px: NDArray[np.float64]  # (n_cameras,)

    @classmethod
    def from_cameras(cls, cameras: Mapping[int, CameraData]) -> _Rig:
        cam_ids = tuple(sorted(cameras))
        projections = np.empty((len(cam_ids), 3, 4))
        focal_px = np.empty(len(cam_ids))
        for index, cam_id in enumerate(cam_ids):
            camera = cameras[cam_id]
            assert camera.rotation is not None and camera.translation is not None and camera.matrix is not None
            projections[index, :, :3] = camera.rotation
            projections[index, :, 3] = np.asarray(camera.translation, dtype=float).reshape(3)
            focal_px[index] = float(camera.matrix[0, 0])
        return cls(cam_ids=cam_ids, projections=projections, focal_px=focal_px)

    @property
    def cam_index(self) -> dict[int, int]:
        return {cam_id: index for index, cam_id in enumerate(self.cam_ids)}


@dataclass(frozen=True, eq=False)
class _Track:
    """One track's undistorted observations, sorted by a shared (sync_index, keypoint_id) key."""

    key: _TrackKey
    cam_index: int
    point_keys: NDArray[np.int64]
    sync_indices: NDArray[np.int64]
    keypoint_ids: NDArray[np.int64]
    xy: NDArray[np.float64]  # normalized image coordinates
    moments: frozenset[int]


def _link_tracks(
    points: ImagePoints,
    cameras: CameraArray,
    box_df: pd.DataFrame,
    track_moments: Mapping[_TrackKey, frozenset[int]],
    max_reprojection_px: float,
) -> _LinkResult:
    """Group unlinked tracks into objects that agree geometrically across cameras.

    Every pair of tracks in different cameras gets a weight from its pooled
    epipolar distance: positive under MAX_PAIR_DISTANCE, negative over it.
    Tracks are then clustered to make the summed weight inside clusters large,
    so a track joins an object on its agreement with all the members at once.

    Only cameras with extrinsics take part; tracks elsewhere are unresolved.
    Two tracks in one camera overlap in time when their track_moments
    intersect, and overlapping tracks never join one object. Callers validate
    the points (unlinked, with track_id) and the cameras' intrinsics.
    max_reprojection_px only sets which observations the reported support
    counts as inliers.
    """
    df = points.df
    posed = cameras.posed_cameras
    observed = df[df["cam_id"].isin(list(posed))]
    rig = _Rig.from_cameras({int(c): posed[int(c)] for c in observed["cam_id"].unique().tolist()})
    tracks = _index_tracks(observed, posed, rig)
    keys = sorted(tracks)
    overlaps = _overlapping_tracks(tracks, track_moments)
    box_sizes = _box_sizes(tracks, box_df)

    weights = np.zeros((len(keys), len(keys)))
    too_few_moments = np.zeros((len(keys), len(keys)), dtype=bool)
    for a, b in combinations(range(len(keys)), 2):
        if keys[a][0] == keys[b][0]:
            continue
        distance, n_moments = _pair_distance(
            tracks[keys[a]], tracks[keys[b]], box_sizes[keys[a]], box_sizes[keys[b]], rig
        )
        if n_moments >= MIN_SUPPORT_MOMENTS:
            weights[a, b] = weights[b, a] = float(np.clip(1.0 - distance / MAX_PAIR_DISTANCE, -1.0, 1.0))
        elif n_moments > 0 and distance <= MAX_PAIR_DISTANCE:
            too_few_moments[a, b] = too_few_moments[b, a] = True

    cannot_link = np.array([[other in overlaps[key] for other in keys] for key in keys], dtype=bool)
    clusters, rivals = _cluster_tracks(weights, cannot_link)

    objects: list[frozenset[_TrackKey]] = []
    member_support: dict[_TrackKey, _Support] = {}
    object_of: dict[int, int] = {}
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        members = frozenset(keys[index] for index in cluster)
        for index in cluster:
            object_of[index] = len(objects)
            rest = [tracks[other] for other in sorted(members) if other != keys[index]]
            member_support[keys[index]] = _support(tracks[keys[index]], rest, rig, max_reprojection_px)
        objects.append(members)

    unresolved: dict[_TrackKey, _Unresolved] = {}
    for key in _moments_by_track(df):
        if key[0] not in posed:
            unresolved[key] = _Unresolved(NO_EXTRINSICS)
    for index, key in enumerate(keys):
        if index in object_of:
            continue
        near = sorted(
            {object_of[int(other)] for other in np.flatnonzero(too_few_moments[index]) if int(other) in object_of}
        )
        if index in rivals:
            unresolved[key] = _Unresolved(CONFLICTING_TRACK, other_track_id=keys[rivals[index]][1])
        elif near:
            unresolved[key] = _Unresolved(TOO_LITTLE_SUPPORT, object_index=near[0])
        else:
            unresolved[key] = _Unresolved(NO_GEOMETRIC_SUPPORT)

    logger.info(f"Linked {len(object_of)} of {len(tracks)} tracks into {len(objects)} objects")
    return _LinkResult(objects=objects, member_support=member_support, unresolved=unresolved)


def _cluster_tracks(
    weights: NDArray[np.float64], cannot_link: NDArray[np.bool_]
) -> tuple[list[list[int]], dict[int, int]]:
    """Merge tracks into clusters, least ambiguous merge first.

    The gain of merging two clusters is the summed weight between them. A
    merge competes with every merge it excludes: one that would put a
    cannot-link track into the same cluster. A merge waits while a competitor
    has at least AMBIGUITY_RATIO of its gain, and among the rest the one with
    the largest lead over its best competitor goes first. Clusters that grow
    can settle a merge that was ambiguous between two tracks alone.

    Returns the clusters as lists of track indices, and for each track left
    waiting at the end, the index of the cannot-link track it competed with.
    """
    clusters = [[index] for index in range(len(weights))]
    gain = weights.copy()
    blocked = cannot_link.copy()
    while True:
        n = len(clusters)
        allowed = (gain > 0) & ~blocked
        # rival[i, j]: best gain of a merge (i, k) where k cannot share a cluster with j.
        possible = np.where(blocked, -np.inf, gain)
        offered = np.where(blocked[None, :, :], possible[:, :, None], -np.inf)
        rival_k = offered.argmax(axis=1) if n else np.zeros((0, 0), dtype=np.intp)
        rival = offered.max(axis=1) if n else np.zeros((0, 0))
        strongest = np.maximum(rival, rival.T)
        waiting = allowed & (strongest >= AMBIGUITY_RATIO * gain)
        ready = np.triu(allowed & ~waiting, k=1)
        if not ready.any():
            break
        lead = np.where(ready, gain - np.maximum(strongest, 0.0), -np.inf)
        i, j = np.unravel_index(int(lead.argmax()), lead.shape)
        clusters[i] = sorted(clusters[i] + clusters[j])
        del clusters[j]
        gain[i, :] += gain[j, :]
        gain[:, i] += gain[:, j]
        gain[i, i] = 0.0
        blocked[i, :] |= blocked[j, :]
        blocked[:, i] |= blocked[:, j]
        gain = np.delete(np.delete(gain, j, axis=0), j, axis=1)
        blocked = np.delete(np.delete(blocked, j, axis=0), j, axis=1)

    rivals: dict[int, int] = {}
    for i, j in zip(*np.nonzero(waiting)):
        # The single track of cluster j competes with a track of cluster rival_k[i, j] for cluster i.
        if rival[i, j] >= AMBIGUITY_RATIO * gain[i, j] and len(clusters[j]) == 1:
            competitor = clusters[int(rival_k[i, j])]
            track = clusters[j][0]
            rivals.setdefault(track, next(other for other in competitor if cannot_link[track, other]))
    return clusters, rivals


def _box_sizes(tracks: Mapping[_TrackKey, _Track], box_df: pd.DataFrame) -> dict[_TrackKey, NDArray[np.float64]]:
    """Square root of box area for each observation of each track, from the track's box at that moment.

    A moment without a box takes the track's median size.
    """
    corners = box_df[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(dtype=float)
    indexed = pd.Series(
        np.sqrt((corners[:, 2] - corners[:, 0]) * (corners[:, 3] - corners[:, 1])),
        index=pd.MultiIndex.from_arrays([box_df["cam_id"], box_df["track_id"], box_df["sync_index"]]),
    )
    indexed = indexed[~indexed.index.duplicated()]
    box_sizes: dict[_TrackKey, NDArray[np.float64]] = {}
    for key, track in tracks.items():
        of_track = indexed.xs(key, level=(0, 1))
        at_moments = of_track.reindex(track.sync_indices).to_numpy(dtype=float)
        box_sizes[key] = np.where(np.isfinite(at_moments), at_moments, float(of_track.median()))
    return box_sizes


def _pair_distance(
    track_a: _Track,
    track_b: _Track,
    box_size_a: NDArray[np.float64],
    box_size_b: NDArray[np.float64],
    rig: _Rig,
) -> tuple[float, int]:
    """Pooled epipolar disagreement of two tracks in different cameras.

    For every (sync_index, keypoint_id) both tracks hold, take the distance
    from each observation to the epipolar line of the other, in pixels, divide
    it by the size of its own track's box at that moment, and average the two.
    Returns the mean over all shared observations, and the number of moments
    with at least MIN_KEYPOINTS_PER_MOMENT shared keypoints. The distance is
    inf when nothing is shared.
    """
    _, rows_a, rows_b = np.intersect1d(track_a.point_keys, track_b.point_keys, assume_unique=True, return_indices=True)
    if len(rows_a) == 0:
        return float("inf"), 0

    P_a, P_b = rig.projections[track_a.cam_index], rig.projections[track_b.cam_index]
    rotation = P_b[:, :3] @ P_a[:, :3].T
    translation = P_b[:, 3] - rotation @ P_a[:, 3]
    cross = np.array(
        [
            [0.0, -translation[2], translation[1]],
            [translation[2], 0.0, -translation[0]],
            [-translation[1], translation[0], 0.0],
        ]
    )
    essential = cross @ rotation

    xy_a = np.column_stack([track_a.xy[rows_a], np.ones(len(rows_a))])
    xy_b = np.column_stack([track_b.xy[rows_b], np.ones(len(rows_b))])
    lines_in_b = xy_a @ essential.T
    lines_in_a = xy_b @ essential
    algebraic = np.abs(np.einsum("ni,ni->n", xy_b, lines_in_b))
    with np.errstate(divide="ignore", invalid="ignore"):
        px_a = rig.focal_px[track_a.cam_index] * algebraic / np.linalg.norm(lines_in_a[:, :2], axis=1)
        px_b = rig.focal_px[track_b.cam_index] * algebraic / np.linalg.norm(lines_in_b[:, :2], axis=1)
    relative = 0.5 * (px_a / box_size_a[rows_a] + px_b / box_size_b[rows_b])
    relative = relative[np.isfinite(relative)]
    if len(relative) == 0:
        return float("inf"), 0

    _, per_moment = np.unique(track_a.sync_indices[rows_a], return_counts=True)
    return float(relative.mean()), int((per_moment >= MIN_KEYPOINTS_PER_MOMENT).sum())


def _index_tracks(df: pd.DataFrame, cameras: Mapping[int, CameraData], rig: _Rig) -> dict[_TrackKey, _Track]:
    """Undistort every observation once and split them into per-track arrays."""
    df = df.drop_duplicates(subset=["cam_id", "track_id", "sync_index", "keypoint_id"])
    df = df.assign(point_key=df.groupby(["sync_index", "keypoint_id"], sort=True).ngroup())

    normalized = np.empty((len(df), 2))
    cam_column = df["cam_id"].to_numpy()
    for cam_id in rig.cam_ids:
        mask = cam_column == cam_id
        pixels = df.loc[mask, ["img_loc_x", "img_loc_y"]].to_numpy(dtype=float)
        normalized[mask] = cameras[cam_id].undistort_points(pixels, output="normalized").astype(float)
    df = df.assign(x_normalized=normalized[:, 0], y_normalized=normalized[:, 1])

    cam_index = rig.cam_index
    tracks: dict[_TrackKey, _Track] = {}
    for _, rows in df.groupby(["cam_id", "track_id"], sort=True):
        rows = rows.sort_values("point_key")
        key = (int(rows["cam_id"].to_numpy()[0]), int(rows["track_id"].to_numpy()[0]))
        sync_indices = rows["sync_index"].to_numpy(dtype=np.int64)
        tracks[key] = _Track(
            key=key,
            cam_index=cam_index[key[0]],
            point_keys=rows["point_key"].to_numpy(dtype=np.int64),
            sync_indices=sync_indices,
            keypoint_ids=rows["keypoint_id"].to_numpy(dtype=np.int64),
            xy=rows[["x_normalized", "y_normalized"]].to_numpy(dtype=float),
            moments=frozenset(int(s) for s in sync_indices),
        )
    return tracks


def _overlapping_tracks(
    tracks: Mapping[_TrackKey, _Track], track_moments: Mapping[_TrackKey, frozenset[int]]
) -> dict[_TrackKey, frozenset[_TrackKey]]:
    """For each track, the other tracks in its camera whose moments intersect its own."""
    overlaps: dict[_TrackKey, set[_TrackKey]] = {key: set() for key in tracks}
    for keys in _tracks_by_camera(tracks).values():
        for key_a, key_b in combinations(keys, 2):
            if track_moments.get(key_a, frozenset()) & track_moments.get(key_b, frozenset()):
                overlaps[key_a].add(key_b)
                overlaps[key_b].add(key_a)
    return {key: frozenset(others) for key, others in overlaps.items()}


def _tracks_by_camera(tracks: Iterable[_TrackKey]) -> dict[int, list[_TrackKey]]:
    by_camera: dict[int, list[_TrackKey]] = {}
    for key in sorted(tracks):
        by_camera.setdefault(key[0], []).append(key)
    return by_camera


def _pairwise_seeds(
    tracks: Mapping[_TrackKey, _Track], rig: _Rig, max_reprojection_px: float
) -> dict[tuple[_TrackKey, _TrackKey], _Support]:
    """Two-view support of every cross-camera track pair that shares enough sampled moments.

    The linker no longer seeds objects from pairs. This remains for the test
    that ranks pairs by ray angle.
    """
    by_camera = _tracks_by_camera(tracks)
    seeds: dict[tuple[_TrackKey, _TrackKey], _Support] = {}
    for cam_a, cam_b in combinations(sorted(by_camera), 2):
        for key_a in by_camera[cam_a]:
            for key_b in by_camera[cam_b]:
                if len(tracks[key_a].moments & tracks[key_b].moments) < MIN_SHARED_MOMENTS:
                    continue
                support = _support(tracks[key_b], [tracks[key_a]], rig, max_reprojection_px)
                if support.passes:
                    seeds[(key_a, key_b)] = support
    return seeds


###############################################################################
# Geometry
###############################################################################


def _support(candidate: _Track, members: Sequence[_Track], rig: _Rig, max_reprojection_px: float) -> _Support:
    """Score a candidate track against member tracks in other cameras.

    For each (sync_index, keypoint_id) the candidate shares with the members,
    triangulate the member views and the candidate's view together and
    reproject into every view. An observation is an inlier when the largest
    residual is within max_reprojection_px and the point is in front of every
    camera used. Weakly crossing rays do not exclude an observation; they only
    lower n_conditioned_moments, which orders candidates.
    """
    others = [member for member in members if member.key[0] != candidate.key[0]]
    n_rows = len(candidate.point_keys)
    if not others or n_rows == 0:
        return _NO_SUPPORT

    member_keys = np.concatenate([member.point_keys for member in others])
    member_cams = np.concatenate([np.full(len(member.point_keys), member.cam_index) for member in others])
    member_xy = np.concatenate([member.xy for member in others])

    position = np.minimum(np.searchsorted(candidate.point_keys, member_keys), n_rows - 1)
    shared = candidate.point_keys[position] == member_keys
    rows = position[shared]
    member_cams = member_cams[shared]
    member_xy = member_xy[shared]

    n_member_views = np.bincount(rows, minlength=n_rows)
    evaluated = n_member_views > 0
    if not evaluated.any():
        return _NO_SUPPORT
    evaluated_rows = np.flatnonzero(evaluated)
    candidate_cams = np.full(n_rows, candidate.cam_index)

    world = _triangulate(
        np.concatenate([rows, evaluated_rows]),
        np.concatenate([member_cams, candidate_cams[evaluated_rows]]),
        np.concatenate([member_xy, candidate.xy[evaluated_rows]]),
        n_rows,
        rig.projections,
    )
    candidate_px, candidate_depth = _reproject(world, candidate_cams, candidate.xy, rig)
    member_px, member_depth = _reproject(world[rows], member_cams, member_xy, rig)

    member_behind = np.bincount(rows, weights=(~(member_depth > 0)).astype(float), minlength=n_rows) > 0
    worst_member_px = np.zeros(n_rows)
    np.maximum.at(worst_member_px, rows, member_px)

    residual = np.maximum(candidate_px, worst_member_px)
    residual = np.where(np.isfinite(residual), residual, np.inf)
    inlier = evaluated & ~member_behind & (candidate_depth > 0) & (residual <= max_reprojection_px)

    ray_angle = _ray_angles(rows, member_cams, member_xy, candidate_cams, candidate.xy, rig)
    conditioned = inlier & (ray_angle >= MIN_RAY_ANGLE_DEG)

    _, per_moment = np.unique(candidate.sync_indices[inlier], return_counts=True)
    _, per_moment_conditioned = np.unique(candidate.sync_indices[conditioned], return_counts=True)
    evaluated_ids, evaluated_counts = np.unique(candidate.keypoint_ids[evaluated], return_counts=True)
    inlier_ids, inlier_counts = np.unique(candidate.keypoint_ids[inlier], return_counts=True)
    inliers_by_keypoint = dict(zip(inlier_ids.tolist(), inlier_counts.tolist()))

    return _Support(
        n_moments=int((per_moment >= MIN_KEYPOINTS_PER_MOMENT).sum()),
        n_conditioned_moments=int((per_moment_conditioned >= MIN_KEYPOINTS_PER_MOMENT).sum()),
        n_observations=int(inlier.sum()),
        n_evaluated=int(evaluated.sum()),
        median_px=float(np.median(residual[evaluated])),
        keypoint_counts={
            int(k): (int(n), int(inliers_by_keypoint.get(k, 0)))
            for k, n in zip(evaluated_ids.tolist(), evaluated_counts.tolist())
        },
    )


def _ray_angles(
    rows: NDArray[np.int64],
    member_cams: NDArray[np.int64],
    member_xy: NDArray[np.float64],
    candidate_cams: NDArray[np.int64],
    candidate_xy: NDArray[np.float64],
    rig: _Rig,
) -> NDArray[np.float64]:
    """Per candidate row, the largest line angle in degrees between its ray and any member ray at that key.

    0 means the rays lie on one line, 90 means perpendicular. Rows without a
    member view get 0. Needs no triangulation.
    """
    candidate_dirs = _unit_rays(candidate_cams, candidate_xy, rig)
    member_dirs = _unit_rays(member_cams, member_xy, rig)
    cosine = np.abs(np.einsum("ni,ni->n", member_dirs, candidate_dirs[rows]))
    angles = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
    largest = np.zeros(len(candidate_cams))
    np.maximum.at(largest, rows, angles)
    return largest


def _unit_rays(cam_indices: NDArray[np.int64], xy: NDArray[np.float64], rig: _Rig) -> NDArray[np.float64]:
    """World-frame unit direction of each normalized observation's back-projected ray."""
    rotation = rig.projections[cam_indices][:, :, :3]
    homogeneous = np.column_stack([xy, np.ones(len(xy))])
    directions = np.einsum("nji,nj->ni", rotation, homogeneous)
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _triangulate(
    point_rows: NDArray[np.int64],
    cam_indices: NDArray[np.int64],
    xy: NDArray[np.float64],
    n_points: int,
    projections: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Linear (DLT) triangulation of many points at once from normalized observations.

    Each view contributes its two DLT rows to its point's 4x4 normal matrix; the
    point is the eigenvector of the smallest eigenvalue. Points may have any
    number of views. Returns (n_points, 3), NaN where a point has no finite
    solution (including points with no views).
    """
    P = projections[cam_indices]
    rows_x = xy[:, 0:1] * P[:, 2, :] - P[:, 0, :]
    rows_y = xy[:, 1:2] * P[:, 2, :] - P[:, 1, :]
    per_view = rows_x[:, :, None] * rows_x[:, None, :] + rows_y[:, :, None] * rows_y[:, None, :]
    normal = np.zeros((n_points, 4, 4))
    np.add.at(normal, point_rows, per_view)

    _, eigenvectors = np.linalg.eigh(normal)
    homogeneous = eigenvectors[:, :, 0]
    w = homogeneous[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        world = homogeneous[:, :3] / w[:, None]
    world[np.abs(w) < 1e-12] = np.nan
    return world


def _reproject(
    world: NDArray[np.float64], cam_indices: NDArray[np.int64], xy: NDArray[np.float64], rig: _Rig
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Residual in each observing camera's pixels and depth in that camera, per observation."""
    P = rig.projections[cam_indices]
    in_camera = np.einsum("nij,nj->ni", P[:, :, :3], world) + P[:, :, 3]
    depth = in_camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = in_camera[:, :2] / depth[:, None]
    residual_px = rig.focal_px[cam_indices] * np.linalg.norm(projected - xy, axis=1)
    return residual_px, depth
