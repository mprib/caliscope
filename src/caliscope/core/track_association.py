"""Link camera-local tracks into cross-camera objects.

Every track starts unlinked (negative object_id). associate_tracks decides which
tracks in different cameras show the same object, from known extrinsics and
re-identification embeddings, and returns the decision as TrackAssociations.
BoxTracks.apply_associations then writes the linked object IDs onto the boxes.

The decision has two stages. At each sampled moment the boxes are matched across
cameras with the method of Dong et al. (2019), in moment_matching. The matches
are then counted for every pair of tracks over all moments, and the tracks are
clustered on those counts. The second stage is Caliscope's own: the published
method works on one moment at a time.

Without extrinsics, tracks cannot be linked. BoxTracks.primary_tracks picks one
track per camera by bounding box footprint instead.
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
from caliscope.core.moment_matching import (
    fuse_affinities,
    geometric_affinity,
    match_moment,
    reidentification_affinity,
)
from caliscope.core.point_data import ImagePoints, _validate_dataframe
from caliscope.core.reidentification_embeddings import ReidentificationEmbeddings
from caliscope.exceptions import CalibrationError
from caliscope.persistence import PersistenceError, _safe_write_toml

logger = logging.getLogger(__name__)

# Largest epipolar distance at one moment, as a fraction of box size, at which two boxes can still match.
# Measured on real footage with reference rigs (Pose2Sim multi-person demo, 16 BioCV trials): with
# embeddings, every full rig and every two- and three-camera subset linked correctly for values from
# 0.08 to 0.60. Below 0.08 a pair of cameras facing each other is missed.
MAX_PAIR_DISTANCE = 0.20
# Shared moments two tracks need before their match count is evidence for or against a link.
MIN_SUPPORT_MOMENTS = 3
# Shared keypoints two boxes need at a moment for their distance to count.
MIN_KEYPOINTS_PER_MOMENT = 3
# A competing merge with at least this share of a merge's gain makes it ambiguous.
AMBIGUITY_RATIO = 0.8
# Cosine similarity of two embeddings at which re-identification neither favours nor opposes a match,
# and the similarity at which it favours one fully. Measured with OSNet-AIN x1.0 (MSMT17) on the same
# two sources: boxes of different people score 0.48 and 0.50 on average (99th percentile 0.64 and 0.65),
# boxes of one person 0.75 and 0.83 (1st percentile 0.54 and 0.61), and the best single cuts are 0.59
# and 0.66. The linker was tested on those clips, so these are not held-out values. Another encoder
# needs its own.
NEUTRAL_SIMILARITY = 0.62
FULL_SIMILARITY = 0.74

NO_KEYPOINTS_SAMPLED = "no keypoints sampled"
NO_EXTRINSICS = "camera has no extrinsics"
NO_MATCH = "matched no track in another camera"
CONFLICTING_TRACK = "conflicts with track {other} in the same camera"
TOO_LITTLE_SUPPORT = "too few shared moments to join object {object_id}"

LINKED_COLUMNS: dict[str, dict[str, object]] = {
    "object_id": {"dtype": "int", "nullable": False},
    "cam_id": {"dtype": "int", "nullable": False},
    "track_id": {"dtype": "int", "nullable": False},
    "n_moments": {"dtype": "int", "nullable": True},
    "agreement": {"dtype": "float", "nullable": True},
}

UNRESOLVED_COLUMNS: dict[str, dict[str, object]] = {
    "cam_id": {"dtype": "int", "nullable": False},
    "track_id": {"dtype": "int", "nullable": False},
    "reason": {"dtype": "str", "nullable": False},
}

PAIR_EVIDENCE_COLUMNS: dict[str, dict[str, object]] = {
    "cam_a": {"dtype": "int", "nullable": False},
    "track_a": {"dtype": "int", "nullable": False},
    "cam_b": {"dtype": "int", "nullable": False},
    "track_b": {"dtype": "int", "nullable": False},
    "n_opportunities": {"dtype": "int", "nullable": False},
    "n_matches": {"dtype": "int", "nullable": False},
}

OBJECT_COLUMNS: dict[str, dict[str, object]] = {
    "object_id": {"dtype": "int", "nullable": False},
    "detector_score": {"dtype": "float", "nullable": True},
    "keypoint_confidence": {"dtype": "float", "nullable": True},
}

_TrackKey = tuple[int, int]
_TrackPair = tuple[_TrackKey, _TrackKey]


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

    linked has one row per linked track with its object_id (0 and up). When
    produced by associate_tracks it also has n_moments, the sampled moments at
    which the track was matched to another track of its object, and agreement,
    the share of its chances to match those tracks that did match.

    unresolved has one row per track that was not linked, with a reason.

    pair_evidence has one row per pair of tracks in different cameras that
    were seen together: n_opportunities, the sampled moments both were seen
    at, and n_matches, the moments the matching put them together.

    objects has one row per linked object with the median detector score of
    its boxes and the median confidence of its sampled keypoints. Low values
    suggest the detector tracked something that is not what it was trained on.

    A (cam_id, track_id) appears at most once across linked and unresolved.
    """

    _linked: pd.DataFrame
    _unresolved: pd.DataFrame = field(default_factory=lambda: _empty_table(UNRESOLVED_COLUMNS))
    _pair_evidence: pd.DataFrame = field(default_factory=lambda: _empty_table(PAIR_EVIDENCE_COLUMNS))
    _objects: pd.DataFrame = field(default_factory=lambda: _empty_table(OBJECT_COLUMNS))

    def __post_init__(self) -> None:
        linked = _validated_table(
            self._linked, LINKED_COLUMNS, "TrackAssociations.linked", ["object_id", "cam_id", "track_id"]
        )
        unresolved = _validated_table(
            self._unresolved, UNRESOLVED_COLUMNS, "TrackAssociations.unresolved", ["cam_id", "track_id"]
        )
        pair_evidence = _validated_table(
            self._pair_evidence,
            PAIR_EVIDENCE_COLUMNS,
            "TrackAssociations.pair_evidence",
            ["cam_a", "track_a", "cam_b", "track_b"],
        )
        objects = _validated_table(self._objects, OBJECT_COLUMNS, "TrackAssociations.objects", ["object_id"])

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
        object.__setattr__(self, "_pair_evidence", pair_evidence)
        object.__setattr__(self, "_objects", objects)

    @property
    def linked(self) -> pd.DataFrame:
        """Return a copy of the linked-track table."""
        return self._linked.copy()

    @property
    def unresolved(self) -> pd.DataFrame:
        """Return a copy of the unresolved-track table."""
        return self._unresolved.copy()

    @property
    def pair_evidence(self) -> pd.DataFrame:
        """Return a copy of the per-pair match counts."""
        return self._pair_evidence.copy()

    @property
    def objects(self) -> pd.DataFrame:
        """Return a copy of the per-object summary."""
        return self._objects.copy()

    @classmethod
    def from_mapping(cls, mapping: Mapping[int, Mapping[int, Iterable[int]]]) -> TrackAssociations:
        """Link tracks by hand: {object_id: {cam_id: [track_ids]}}.

        Evidence is left empty and there are no unresolved rows.

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
        """Write one TOML file atomically. An empty value is an omitted key.

        Raises:
            PersistenceError: if the file cannot be written.
        """
        path = Path(path)
        data: dict[str, list[dict[str, int | float | str]]] = {}
        for section, table in (
            ("linked", self._linked),
            ("unresolved", self._unresolved),
            ("pair_evidence", self._pair_evidence),
            ("object", self._objects),
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
                pd.DataFrame(data.get("pair_evidence", []), columns=list(PAIR_EVIDENCE_COLUMNS)),
                pd.DataFrame(data.get("object", []), columns=list(OBJECT_COLUMNS)),
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
    reidentification_embeddings: ReidentificationEmbeddings,
) -> TrackAssociations:
    """Link camera-local tracks into cross-camera objects with fixed extrinsics.

    unlinked_points and reidentification_embeddings come from the same sampled
    boxes (BoxTracks.sample_for_association), through estimate_keypoints and
    extract_reidentification_embeddings. boxes are the full-rate boxes.

    At each sampled moment the boxes are matched across cameras. Two boxes can
    match when their keypoints lie near each other's epipolar lines, as a
    fraction of box size (MAX_PAIR_DISTANCE), and their embeddings are alike.
    The matches of all cameras at a moment are made to agree with each other.
    Each pair of tracks is then scored by how often it matched out of the
    moments both were seen, and tracks are merged into objects while the scores
    between them are positive on balance.

    Every track in boxes appears once in the result, either linked or
    unresolved with a reason. A track belongs to at most one object, and the
    tracks of one object never overlap in time within a camera. When two
    overlapping tracks in a camera fit an object about equally well, both stay
    unresolved. Objects are numbered by box footprint summed across cameras,
    largest first, using the full-rate boxes. Camera parameters are never
    changed.

    Raises:
        ValueError: if track_id is missing or null, if any row is already
            linked (object_id >= 0), if points name a (cam_id, track_id)
            absent from boxes, or if no embedding belongs to a sampled box of
            the points.
        CalibrationError: if a camera in the points lacks intrinsics, or if
            fewer than two cameras in the points have extrinsics. The message
            points at bootstrap_extrinsics.
    """
    _validate_association_inputs(unlinked_points, cameras, boxes, reidentification_embeddings)

    box_df = boxes.df
    point_df = unlinked_points.df
    track_moments = _moments_by_track(box_df)

    result = _link_tracks(unlinked_points, cameras, box_df, track_moments, reidentification_embeddings)
    object_ids = _rank_by_footprint(result.objects, box_df)

    linked_rows: list[dict[str, int | float | None]] = []
    object_rows: list[dict[str, int | float | None]] = []
    linked_keys: set[_TrackKey] = set()
    for index, members in enumerate(result.objects):
        for key in sorted(members):
            n_moments, agreement = _member_evidence(key, members, result)
            linked_keys.add(key)
            linked_rows.append(
                {
                    "object_id": object_ids[index],
                    "cam_id": key[0],
                    "track_id": key[1],
                    "n_moments": n_moments,
                    "agreement": agreement,
                }
            )
        object_rows.append(
            {
                "object_id": object_ids[index],
                "detector_score": _median_of_tracks(box_df, members, "detector_score"),
                "keypoint_confidence": _median_of_tracks(point_df, members, "confidence"),
            }
        )

    pair_rows = [
        {
            "cam_a": key_a[0],
            "track_a": key_a[1],
            "cam_b": key_b[0],
            "track_b": key_b[1],
            "n_opportunities": len(moments),
            "n_matches": len(result.matches.get((key_a, key_b), ())),
        }
        for (key_a, key_b), moments in sorted(result.opportunities.items())
    ]

    tracks_with_points = set(_moments_by_track(point_df))
    posed = cameras.posed_cameras
    unresolved_rows: list[dict[str, int | str]] = []
    for key in sorted(track_moments):
        if key in linked_keys:
            continue
        if key[0] not in posed:
            reason = NO_EXTRINSICS
        elif key not in tracks_with_points:
            reason = NO_KEYPOINTS_SAMPLED
        else:
            reason = result.unresolved.get(key, _Unresolved(NO_MATCH)).describe(object_ids)
        unresolved_rows.append({"cam_id": key[0], "track_id": key[1], "reason": reason})

    logger.info(
        f"associate_tracks: linked {len(linked_rows)} tracks into {len(result.objects)} objects; "
        f"{len(unresolved_rows)} tracks unresolved"
    )
    return TrackAssociations(
        pd.DataFrame(linked_rows, columns=list(LINKED_COLUMNS)),
        pd.DataFrame(unresolved_rows, columns=list(UNRESOLVED_COLUMNS)),
        pd.DataFrame(pair_rows, columns=list(PAIR_EVIDENCE_COLUMNS)),
        pd.DataFrame(object_rows, columns=list(OBJECT_COLUMNS)),
    )


def _median_of_tracks(df: pd.DataFrame, members: Iterable[_TrackKey], column: str) -> float | None:
    """Median of a column over the rows of the given tracks, or None when there is nothing to take it over."""
    if column not in df.columns:
        return None
    keys = pd.MultiIndex.from_arrays([df["cam_id"], df["track_id"]])
    values = df.loc[keys.isin(list(members)), column].dropna()
    return float(values.median()) if len(values) else None


def _member_evidence(key: _TrackKey, members: Iterable[_TrackKey], result: _LinkResult) -> tuple[int, float | None]:
    """Moments at which a track matched another member, and the share of its chances with members that matched."""
    n_opportunities = n_matches = 0
    matched_moments: set[int] = set()
    for other in members:
        if other == key:
            continue
        pair = (key, other) if key < other else (other, key)
        matched = result.matches.get(pair, frozenset())
        n_opportunities += len(result.opportunities.get(pair, ()))
        n_matches += len(matched)
        matched_moments |= matched
    return len(matched_moments), n_matches / n_opportunities if n_opportunities else None


def _validate_association_inputs(
    points: ImagePoints, cameras: CameraArray, boxes: BoxTracks, embeddings: ReidentificationEmbeddings
) -> None:
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

    sampled = pd.MultiIndex.from_frame(df[["sync_index", "cam_id", "track_id"]].drop_duplicates())
    if not sampled.isin(pd.MultiIndex.from_frame(embeddings.index)).any():
        raise ValueError(
            "reidentification_embeddings hold no embedding for any sampled box of the points; run "
            "extract_reidentification_embeddings on the same sampled boxes as estimate_keypoints"
        )

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
class _Unresolved:
    reason: str
    other_track_id: int | None = None
    object_index: int | None = None

    def describe(self, object_ids: Sequence[int]) -> str:
        object_id = object_ids[self.object_index] if self.object_index is not None else None
        return self.reason.format(other=self.other_track_id, object_id=object_id)


@dataclass(frozen=True)
class _LinkResult:
    """Objects as sets of (cam_id, track_id), with the match counts of every cross-camera track pair.

    opportunities and matches hold, per pair, the sampled moments both tracks
    were seen at and the moments the matching put them together. Pairs are
    keyed smaller track first.
    """

    objects: list[frozenset[_TrackKey]]
    unresolved: dict[_TrackKey, _Unresolved]
    opportunities: dict[_TrackPair, frozenset[int]]
    matches: dict[_TrackPair, frozenset[int]]


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
    embeddings: ReidentificationEmbeddings,
) -> _LinkResult:
    """Group unlinked tracks into objects from how often they match at single moments.

    Each pair of tracks in different cameras gets a weight from its match
    count: 1 when it matched at every moment both were seen, -1 when it never
    did, and 0 when it has fewer than MIN_SUPPORT_MOMENTS shared moments.
    Tracks are then clustered to make the summed weight inside clusters large,
    so a track joins an object on its agreement with all the members at once.

    Only cameras with extrinsics take part; tracks elsewhere are unresolved.
    Two tracks in one camera overlap in time when their track_moments
    intersect, and overlapping tracks never join one object. Callers validate
    the points (unlinked, with track_id) and the cameras' intrinsics.
    """
    df = points.df
    posed = cameras.posed_cameras
    observed = df[df["cam_id"].isin(list(posed))]
    rig = _Rig.from_cameras({int(c): posed[int(c)] for c in observed["cam_id"].unique().tolist()})
    tracks = _index_tracks(observed, posed, rig)
    keys = sorted(tracks)
    position = {key: index for index, key in enumerate(keys)}
    overlaps = _overlapping_tracks(tracks, track_moments)
    opportunities, matches, left_out_with = _match_moments(tracks, _box_sizes(tracks, box_df), rig, embeddings)

    weights = np.zeros((len(keys), len(keys)))
    too_few_moments = np.zeros((len(keys), len(keys)), dtype=bool)
    for pair, moments in opportunities.items():
        a, b = position[pair[0]], position[pair[1]]
        n_matches = len(matches.get(pair, ()))
        if len(moments) >= MIN_SUPPORT_MOMENTS:
            weights[a, b] = weights[b, a] = 2.0 * n_matches / len(moments) - 1.0
        elif n_matches:
            too_few_moments[a, b] = too_few_moments[b, a] = True

    cannot_link = np.array([[other in overlaps[key] for other in keys] for key in keys], dtype=bool)
    clusters, rivals = _cluster_tracks(weights, cannot_link)

    objects: list[frozenset[_TrackKey]] = []
    object_of: dict[int, int] = {}
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        for index in cluster:
            object_of[index] = len(objects)
        objects.append(frozenset(keys[index] for index in cluster))

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
        elif key in left_out_with:
            unresolved[key] = _Unresolved(CONFLICTING_TRACK, other_track_id=left_out_with[key][1])
        else:
            unresolved[key] = _Unresolved(NO_MATCH)

    logger.info(f"Linked {len(object_of)} of {len(tracks)} tracks into {len(objects)} objects")
    return _LinkResult(objects=objects, unresolved=unresolved, opportunities=opportunities, matches=matches)


def _match_moments(
    tracks: Mapping[_TrackKey, _Track],
    box_sizes: Mapping[_TrackKey, NDArray[np.float64]],
    rig: _Rig,
    embeddings: ReidentificationEmbeddings,
) -> tuple[dict[_TrackPair, frozenset[int]], dict[_TrackPair, frozenset[int]], dict[_TrackKey, _TrackKey]]:
    """Match the boxes at every sampled moment and collect the outcome per pair of tracks.

    A pair has an opportunity at each moment both tracks share at least
    MIN_KEYPOINTS_PER_MOMENT keypoints, and a match where the matching puts
    them in one group. A box with no embedding is matched on geometry alone.

    Also returns, for each track the matching left out because it and another
    track of its camera fit the same group, the track it was most often left
    out with.
    """
    distances: dict[int, dict[_TrackPair, float]] = {}
    for key_a, key_b in combinations(sorted(tracks), 2):
        if key_a[0] == key_b[0] or not tracks[key_a].moments & tracks[key_b].moments:
            continue
        at_moments = _moment_distances(tracks[key_a], tracks[key_b], box_sizes[key_a], box_sizes[key_b], rig)
        for moment, distance in at_moments.items():
            distances.setdefault(moment, {})[key_a, key_b] = distance

    vectors: dict[tuple[int, _TrackKey], NDArray[np.float32]] = {}
    index = embeddings.index[["sync_index", "cam_id", "track_id"]].to_numpy(dtype=np.int64)
    for (sync_index, cam_id, track_id), vector in zip(index.tolist(), embeddings.vectors):
        vectors[sync_index, (cam_id, track_id)] = vector

    opportunities: dict[_TrackPair, set[int]] = {}
    matches: dict[_TrackPair, set[int]] = {}
    left_out_counts: dict[_TrackKey, dict[_TrackKey, int]] = {}
    for moment in sorted(distances):
        pairs = distances[moment]
        boxes_here = sorted({key for pair in pairs for key in pair})
        position = {key: index for index, key in enumerate(boxes_here)}
        distance = np.full((len(boxes_here), len(boxes_here)), np.inf)
        for (key_a, key_b), value in pairs.items():
            distance[position[key_a], position[key_b]] = distance[position[key_b], position[key_a]] = value
            opportunities.setdefault((key_a, key_b), set()).add(moment)
        affinity = geometric_affinity(distance, MAX_PAIR_DISTANCE)
        for key_a, key_b in pairs:
            vector_a, vector_b = vectors.get((moment, key_a)), vectors.get((moment, key_b))
            if vector_a is None or vector_b is None:
                continue
            a, b = position[key_a], position[key_b]
            similarity = np.array(float(vector_a @ vector_b))
            affinity[a, b] = affinity[b, a] = fuse_affinities(
                affinity[a, b], reidentification_affinity(similarity, NEUTRAL_SIMILARITY, FULL_SIMILARITY)
            )
        views = np.array([tracks[key].cam_index for key in boxes_here], dtype=np.int64)
        matched = match_moment(affinity, views)
        for group in matched.groups:
            for a, b in combinations(group, 2):
                matches.setdefault((boxes_here[a], boxes_here[b]), set()).add(moment)
        for left_out in matched.inconsistent:
            for a, b in combinations(left_out, 2):
                key_a, key_b = boxes_here[a], boxes_here[b]
                if key_a[0] == key_b[0]:
                    for key, other in ((key_a, key_b), (key_b, key_a)):
                        counts = left_out_counts.setdefault(key, {})
                        counts[other] = counts.get(other, 0) + 1

    left_out_with = {
        key: min(counts, key=lambda other: (-counts[other], other)) for key, counts in left_out_counts.items()
    }
    return (
        {pair: frozenset(moments) for pair, moments in opportunities.items()},
        {pair: frozenset(moments) for pair, moments in matches.items()},
        left_out_with,
    )


def _moment_distances(
    track_a: _Track,
    track_b: _Track,
    box_size_a: NDArray[np.float64],
    box_size_b: NDArray[np.float64],
    rig: _Rig,
) -> dict[int, float]:
    """Epipolar disagreement of two tracks in different cameras at each moment they share.

    For every (sync_index, keypoint_id) both tracks hold, take the distance
    from each observation to the epipolar line of the other, in pixels, divide
    it by the size of its own track's box at that moment, and average the two.
    Returns the mean over the keypoints of each moment that has at least
    MIN_KEYPOINTS_PER_MOMENT of them.
    """
    _, rows_a, rows_b = np.intersect1d(track_a.point_keys, track_b.point_keys, assume_unique=True, return_indices=True)
    if len(rows_a) == 0:
        return {}

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
    finite = np.isfinite(relative)
    moments, relative = track_a.sync_indices[rows_a][finite], relative[finite]

    at_moments: dict[int, float] = {}
    for moment in np.unique(moments):
        of_moment = relative[moments == moment]
        if len(of_moment) >= MIN_KEYPOINTS_PER_MOMENT:
            at_moments[int(moment)] = float(of_moment.mean())
    return at_moments


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
