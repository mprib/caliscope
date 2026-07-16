from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import rtoml

from caliscope.core.aruco_marker import ArucoMarkerSet

if TYPE_CHECKING:
    from caliscope.core.charuco import Charuco
    from caliscope.core.chessboard import Chessboard
    from caliscope.core.point_data import ImagePoints


@dataclass(frozen=True)
class DistanceConstraint:
    object_id_a: int
    keypoint_id_a: int
    object_id_b: int
    keypoint_id_b: int
    distance: float
    sigma: float


@dataclass(frozen=True)
class CentroidDistanceConstraint:
    """Distance between two markers' corner centroids.

    A centroid is the mean of the marker's four corner points (keypoint_ids
    0..3 — the ArUco corner convention). Only compiled from center
    DistanceLinks; charuco never produces these. A centroid constraint pins
    only the separation between the two centroids — it is blind to each
    marker's internal orientation and shape, which relies on that marker's
    own (always-compiled) intra-marker distance constraints to stay pinned.
    """

    object_id_a: int
    object_id_b: int
    distance: float
    sigma: float


@dataclass(frozen=True)
class PointRemap:
    """Rewrites one observed (object_id, keypoint_id) to another's identity.

    Compiled from zero-thickness MirrorPairs: marker B's corner observation
    is remapped to marker A's identity and marker A's baked-in obj_loc, so
    both faces of a thin board contribute to the same triangulated world
    point. obj_loc_x/y/z are marker A's corner coordinates at compile time —
    applying the remap needs nothing but the ConstraintSet itself.
    """

    object_id_from: int
    keypoint_id_from: int
    object_id_to: int
    keypoint_id_to: int
    obj_loc_x: float
    obj_loc_y: float
    obj_loc_z: float


@dataclass(frozen=True)
class ConstraintSet:
    distances: tuple[DistanceConstraint, ...]
    static_object_ids: frozenset[int]
    centroid_distances: tuple[CentroidDistanceConstraint, ...] = ()
    point_remaps: tuple[PointRemap, ...] = ()
    # Set by from_charuco (never by aruco/chessboard compilers): the board's
    # substrate thickness in meters, 0.0 for a thin board. Non-None declares a
    # closed identity universe — the extraction must contain exactly the
    # object_ids this thickness implies ({0}, or {0, 1} when > 0) — which lets
    # calibrate_extrinsics fail loudly when thickness changed between
    # extraction and calibration instead of silently mis-calibrating.
    back_face_thickness_m: float | None = None

    @classmethod
    def from_marker_set(
        cls,
        marker_set: ArucoMarkerSet,
        sigma_m: float = 0.002,
        center_sigma_m: float = 0.005,
    ) -> ConstraintSet:
        """Compile distance constraints from a marker set.

        Emits 6 intra-marker constraints per marker (4 edges + 2 diagonals),
        skipped for a marker that is the B-side of a zero-thickness
        MirrorPair (its observations are remapped to marker A's identity, so
        its own geometry constraints would reference an object_id with no
        world points). Each corner DistanceLink passes through as exactly
        one DistanceConstraint — the user's measured distance is the
        constraint, no derived pairs. Each center DistanceLink compiles to
        one CentroidDistanceConstraint. A link's own sigma_m wins when set;
        otherwise corner links default to sigma_m and center links default
        to center_sigma_m. Nonzero-thickness MirrorPairs compile to 4
        DistanceConstraints (one per corresponding corner pair) at
        thickness_m; zero-thickness MirrorPairs compile to 4 PointRemaps
        instead. All distances in meters.
        """
        remapped_marker_ids: set[int] = {pair.marker_b for pair in marker_set.mirror_pairs if pair.is_zero_thickness}

        constraints: list[DistanceConstraint] = []

        for marker_id, marker in marker_set.markers.items():
            if marker_id in remapped_marker_ids:
                continue
            corners = marker.corners
            for i in range(4):
                for j in range(i + 1, 4):
                    dist = float(np.linalg.norm(corners[i] - corners[j]))
                    constraints.append(
                        DistanceConstraint(
                            object_id_a=marker_id,
                            keypoint_id_a=i,
                            object_id_b=marker_id,
                            keypoint_id_b=j,
                            distance=dist,
                            sigma=sigma_m,
                        )
                    )

        centroid_constraints: list[CentroidDistanceConstraint] = []
        for link in marker_set.links:
            if link.is_center:
                centroid_constraints.append(
                    CentroidDistanceConstraint(
                        object_id_a=link.marker_a,
                        object_id_b=link.marker_b,
                        distance=link.distance_m,
                        sigma=link.sigma_m if link.sigma_m is not None else center_sigma_m,
                    )
                )
            else:
                assert link.corner_a is not None and link.corner_b is not None  # is_center=False guarantees this
                constraints.append(
                    DistanceConstraint(
                        object_id_a=link.marker_a,
                        keypoint_id_a=link.corner_a,
                        object_id_b=link.marker_b,
                        keypoint_id_b=link.corner_b,
                        distance=link.distance_m,
                        sigma=link.sigma_m if link.sigma_m is not None else sigma_m,
                    )
                )

        point_remaps: list[PointRemap] = []
        for pair in marker_set.mirror_pairs:
            if pair.is_zero_thickness:
                marker_a = marker_set.markers[pair.marker_a]
                for corner_a, corner_b in pair.corner_mapping:
                    obj_loc = marker_a.corners[corner_a]
                    point_remaps.append(
                        PointRemap(
                            object_id_from=pair.marker_b,
                            keypoint_id_from=corner_b,
                            object_id_to=pair.marker_a,
                            keypoint_id_to=corner_a,
                            obj_loc_x=float(obj_loc[0]),
                            obj_loc_y=float(obj_loc[1]),
                            obj_loc_z=float(obj_loc[2]),
                        )
                    )
            else:
                for corner_a, corner_b in pair.corner_mapping:
                    constraints.append(
                        DistanceConstraint(
                            object_id_a=pair.marker_a,
                            keypoint_id_a=corner_a,
                            object_id_b=pair.marker_b,
                            keypoint_id_b=corner_b,
                            distance=pair.thickness_m,
                            sigma=pair.sigma_m if pair.sigma_m is not None else sigma_m,
                        )
                    )

        static_ids = frozenset(
            mid for mid, m in marker_set.markers.items() if m.static and mid not in remapped_marker_ids
        )

        return cls(
            distances=tuple(constraints),
            static_object_ids=static_ids,
            centroid_distances=tuple(centroid_constraints),
            point_remaps=tuple(point_remaps),
        )

    def remap_image_points(self, image_points: ImagePoints) -> ImagePoints:
        """Apply zero-thickness MirrorPair remaps to observed image points.

        Returns the input object unchanged when point_remaps is empty (cheap
        no-op for charuco and plain-aruco paths). Otherwise rewrites the
        identity and obj_loc columns for each remapped observation and
        constructs a fresh ImagePoints (which revalidates).
        """
        if not self.point_remaps:
            return image_points

        from caliscope.core.point_data import ImagePoints

        df = image_points.df
        for r in self.point_remaps:
            mask = (df["object_id"] == r.object_id_from) & (df["keypoint_id"] == r.keypoint_id_from)
            df.loc[mask, "object_id"] = r.object_id_to
            df.loc[mask, "keypoint_id"] = r.keypoint_id_to
            df.loc[mask, "obj_loc_x"] = r.obj_loc_x
            df.loc[mask, "obj_loc_y"] = r.obj_loc_y
            df.loc[mask, "obj_loc_z"] = r.obj_loc_z

        return ImagePoints(df)

    @staticmethod
    def _truss_distance_constraints(
        corners: np.ndarray, spacing: float, sigma_m: float, object_id: int = 0
    ) -> tuple[DistanceConstraint, ...]:
        """Local-truss + extreme-corner-brace distance constraints for a grid.

        corners is (N, 3) in meters; spacing is the grid pitch in meters.
        object_id defaults to 0 (matching the Charuco/Chessboard tracker
        identity scheme: object_id=0, keypoint_id=corner index); the back face
        of a thick two-sided board passes object_id=1.

        Corners are located on the grid by rounding each coordinate to the
        nearest multiple of spacing, so the layout is recovered from geometry
        rather than assumed corner-id order — OpenCV does not guarantee any
        particular charuco corner-id-to-grid-position layout. Points are emitted
        as float32, so exact equality is unsafe; the rounding scales with board
        size and tolerates float32 error (empirically up to ~1e-6 of a square)
        with wide margin below the 0.5-square ambiguity band.

        Constraint density is a local truss (horizontal + vertical neighbor
        edges, both diagonals of every grid cell) plus 6 global braces among
        the 4 extreme board corners. Rationale: nearest-neighbor + diagonal
        distances alone are invariant under a paper fold along any grid
        line — every truss edge and cell diagonal either lies wholly within
        one rigid half of the fold or has an endpoint on the hinge axis, so
        every constrained *distance* is preserved even though corners move.
        The 4-corner braces cross every interior fold line and kill those
        modes. Full pairwise distances (~C(N,2), roughly triple the residual
        rows for a typical board) would add no rigidity information beyond
        what the truss + braces already pin down.
        """
        x_keys = np.round(corners[:, 0] / spacing).astype(np.int64)
        y_keys = np.round(corners[:, 1] / spacing).astype(np.int64)

        edges: list[tuple[int, int]] = []

        # Horizontal neighbors: group by row (y), sort by column (x).
        rows: dict[int, list[tuple[int, int]]] = {}
        for idx, y_key in enumerate(y_keys):
            rows.setdefault(int(y_key), []).append((int(x_keys[idx]), idx))
        for row_points in rows.values():
            row_points.sort()
            for (_, a), (_, b) in zip(row_points, row_points[1:]):
                edges.append((a, b))

        # Vertical neighbors: group by column (x), sort by row (y).
        cols: dict[int, list[tuple[int, int]]] = {}
        for idx, x_key in enumerate(x_keys):
            cols.setdefault(int(x_key), []).append((int(y_keys[idx]), idx))
        for col_points in cols.values():
            col_points.sort()
            for (_, a), (_, b) in zip(col_points, col_points[1:]):
                edges.append((a, b))

        # Cell diagonals: a corner at (x, y) with neighbors at (x+sq, y),
        # (x, y+sq), (x+sq, y+sq) forms a grid cell; emit both diagonals.
        coord_to_idx = {(int(xk), int(yk)): idx for idx, (xk, yk) in enumerate(zip(x_keys, y_keys))}
        for idx, (x_key, y_key) in enumerate(zip(x_keys, y_keys)):
            x_key, y_key = int(x_key), int(y_key)
            right = coord_to_idx.get((x_key + 1, y_key))
            up = coord_to_idx.get((x_key, y_key + 1))
            diagonal = coord_to_idx.get((x_key + 1, y_key + 1))
            if right is not None and up is not None and diagonal is not None:
                edges.append((idx, diagonal))
                edges.append((right, up))

        # Global braces: all 6 pairwise distances among the 4 extreme corners.
        min_x, max_x = int(x_keys.min()), int(x_keys.max())
        min_y, max_y = int(y_keys.min()), int(y_keys.max())
        extreme_corners = [
            coord_to_idx[(min_x, min_y)],
            coord_to_idx[(min_x, max_y)],
            coord_to_idx[(max_x, min_y)],
            coord_to_idx[(max_x, max_y)],
        ]
        edges.extend(combinations(extreme_corners, 2))

        return tuple(
            DistanceConstraint(
                object_id_a=object_id,
                keypoint_id_a=a,
                object_id_b=object_id,
                keypoint_id_b=b,
                distance=float(np.linalg.norm(corners[a] - corners[b])),
                sigma=sigma_m,
            )
            for a, b in edges
        )

    @staticmethod
    def _cross_face_constraints(
        corners: np.ndarray, spacing: float, thickness_m: float, sigma_m: float
    ) -> tuple[DistanceConstraint, ...]:
        """Ties and braces rigidly relating the two faces of a thick board.

        Per-corner ties (front N <-> back N at distance t) pin only the
        magnitude of the inter-face offset: with the back face rigid, any
        lateral offset d with |d| = t satisfies every tie exactly — a
        continuous 2-DoF shear null space. The right-neighbor brace
        (front N <-> back right neighbor at sqrt(s^2 + t^2)) zeroes the
        offset's x-component and the down-neighbor brace its y-component,
        leaving only the offset along the board normal. This is the
        cross-face analog of the in-face fold braces above. Corners are
        located on the grid by the same spacing-rounding as the truss.
        """
        x_keys = np.round(corners[:, 0] / spacing).astype(np.int64)
        y_keys = np.round(corners[:, 1] / spacing).astype(np.int64)
        coord_to_idx = {(int(xk), int(yk)): idx for idx, (xk, yk) in enumerate(zip(x_keys, y_keys))}

        brace_length = float(np.hypot(spacing, thickness_m))
        rows: list[DistanceConstraint] = []
        for idx, (x_key, y_key) in enumerate(zip(x_keys, y_keys)):
            rows.append(
                DistanceConstraint(
                    object_id_a=0,
                    keypoint_id_a=idx,
                    object_id_b=1,
                    keypoint_id_b=idx,
                    distance=thickness_m,
                    sigma=sigma_m,
                )
            )
            right = coord_to_idx.get((int(x_key) + 1, int(y_key)))
            down = coord_to_idx.get((int(x_key), int(y_key) + 1))
            for neighbor in (right, down):
                if neighbor is not None:
                    rows.append(
                        DistanceConstraint(
                            object_id_a=0,
                            keypoint_id_a=idx,
                            object_id_b=1,
                            keypoint_id_b=neighbor,
                            distance=brace_length,
                            sigma=sigma_m,
                        )
                    )
        return tuple(rows)

    @classmethod
    def from_charuco(cls, charuco: Charuco, sigma_m: float = 0.002, thickness_sigma_m: float = 0.0005) -> ConstraintSet:
        """Compile board-geometry distance constraints for BA.

        Front-face corners are object_id 0, matching CharucoTracker's identity
        scheme (keypoint_id = chessboard corner index). When the board has
        substrate thickness (two-sided board on foam core etc.), the back face
        is object_id 1 and three more constraint groups are emitted: the back
        face's own truss, per-corner cross-face ties at the thickness, and
        right/down cross-face braces that kill the tie-only shear null space
        (see _cross_face_constraints).

        thickness_sigma_m is deliberately tighter than sigma_m: the thickness
        is a single caliper measurement, not a print-scale estimate, and the
        cross-face rows are the sole rigid link between the front- and
        back-viewing camera groups. static_object_ids is empty — the board
        moves. Never produces centroids (no separate markers to link).

        Corner ids come from `charuco.board.getChessboardCorners()`, the same
        (N, 3) array CharucoTracker indexes by `keypoint_id`. Truss density and
        the grid-from-coordinates recovery live in _truss_distance_constraints.
        """
        corners = np.asarray(charuco.board.getChessboardCorners())
        square_length = float(charuco.board.getSquareLength())
        constraints = cls._truss_distance_constraints(corners, square_length, sigma_m)
        if charuco.thickness_m > 0:
            back_truss = cls._truss_distance_constraints(corners, square_length, sigma_m, object_id=1)
            assert all(d.object_id_a == 1 and d.object_id_b == 1 for d in back_truss)
            cross_face = cls._cross_face_constraints(corners, square_length, charuco.thickness_m, thickness_sigma_m)
            constraints = constraints + back_truss + cross_face
        return cls(
            distances=constraints,
            static_object_ids=frozenset(),
            centroid_distances=(),
            back_face_thickness_m=charuco.thickness_m,
        )

    @classmethod
    def from_chessboard(cls, chessboard: Chessboard, sigma_m: float = 0.002) -> ConstraintSet:
        """Compile board-geometry distance constraints from a metric chessboard.

        object_id is 0 for every corner, matching ChessboardTracker's identity
        scheme. static_object_ids is empty — the board moves. Same truss density
        and rationale as from_charuco (see _truss_distance_constraints).

        Requires chessboard.square_size_cm to be set: without it
        get_object_points() returns unit-spacing points, and a unit-spacing
        constraint set is a silent scale bug.
        """
        if chessboard.square_size_cm is None:
            raise ValueError(
                "from_chessboard requires square_size_cm to be set; a unit-spacing "
                "constraint set would silently pin the wrong scale."
            )
        corners = chessboard.get_object_points()
        spacing = chessboard.square_size_cm / 100
        constraints = cls._truss_distance_constraints(corners, spacing, sigma_m)
        return cls(distances=constraints, static_object_ids=frozenset(), centroid_distances=())

    def to_toml(self, path: Path) -> None:
        from caliscope.persistence import _safe_write_toml

        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "static_object_ids": sorted(self.static_object_ids),
            "distances": [
                {
                    "object_id_a": d.object_id_a,
                    "keypoint_id_a": d.keypoint_id_a,
                    "object_id_b": d.object_id_b,
                    "keypoint_id_b": d.keypoint_id_b,
                    "distance": d.distance,
                    "sigma": d.sigma,
                }
                for d in self.distances
            ],
        }
        if self.centroid_distances:
            data["centroid_distances"] = [
                {
                    "object_id_a": c.object_id_a,
                    "object_id_b": c.object_id_b,
                    "distance": c.distance,
                    "sigma": c.sigma,
                }
                for c in self.centroid_distances
            ]
        if self.point_remaps:
            data["point_remaps"] = [
                {
                    "object_id_from": r.object_id_from,
                    "keypoint_id_from": r.keypoint_id_from,
                    "object_id_to": r.object_id_to,
                    "keypoint_id_to": r.keypoint_id_to,
                    "obj_loc_x": r.obj_loc_x,
                    "obj_loc_y": r.obj_loc_y,
                    "obj_loc_z": r.obj_loc_z,
                }
                for r in self.point_remaps
            ]
        if self.back_face_thickness_m is not None:
            data["back_face_thickness_m"] = self.back_face_thickness_m
        _safe_write_toml(data, path)

    @classmethod
    def from_toml(cls, path: Path) -> ConstraintSet:
        from caliscope.persistence import PersistenceError

        if not path.exists():
            raise PersistenceError(f"ConstraintSet file not found: {path}")
        try:
            data = rtoml.load(path)
            distances = tuple(
                DistanceConstraint(
                    object_id_a=d["object_id_a"],
                    keypoint_id_a=d["keypoint_id_a"],
                    object_id_b=d["object_id_b"],
                    keypoint_id_b=d["keypoint_id_b"],
                    distance=d["distance"],
                    sigma=d["sigma"],
                )
                for d in data.get("distances", [])
            )
            centroid_distances = tuple(
                CentroidDistanceConstraint(
                    object_id_a=c["object_id_a"],
                    object_id_b=c["object_id_b"],
                    distance=c["distance"],
                    sigma=c["sigma"],
                )
                for c in data.get("centroid_distances", [])
            )
            point_remaps = tuple(
                PointRemap(
                    object_id_from=r["object_id_from"],
                    keypoint_id_from=r["keypoint_id_from"],
                    object_id_to=r["object_id_to"],
                    keypoint_id_to=r["keypoint_id_to"],
                    obj_loc_x=r["obj_loc_x"],
                    obj_loc_y=r["obj_loc_y"],
                    obj_loc_z=r["obj_loc_z"],
                )
                for r in data.get("point_remaps", [])
            )
            static_ids = frozenset(data.get("static_object_ids", []))
            return cls(
                distances=distances,
                static_object_ids=static_ids,
                centroid_distances=centroid_distances,
                point_remaps=point_remaps,
                back_face_thickness_m=data.get("back_face_thickness_m"),
            )
        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(f"Failed to load ConstraintSet from {path}: {e}") from e


@dataclass(frozen=True)
class ConstraintViolation:
    object_id_a: int
    keypoint_id_a: int
    object_id_b: int
    keypoint_id_b: int
    sync_index: int
    expected: float
    actual: float
    # "centroid" violations have no single corner to name; keypoint_id_a and
    # keypoint_id_b are set to -1 for them.
    kind: Literal["corner", "centroid"] = "corner"


@dataclass(frozen=True)
class RigidityReport:
    violations: tuple[ConstraintViolation, ...]

    @cached_property
    def rmse_mm(self) -> float:
        if not self.violations:
            return 0.0
        errors = np.array([v.actual - v.expected for v in self.violations])
        return float(np.sqrt(np.mean(errors**2)) * 1000.0)

    @cached_property
    def relative_rmse_pct(self) -> float:
        if not self.violations:
            return 0.0
        rel_errors = np.array([(v.actual - v.expected) / v.expected for v in self.violations])
        return float(np.sqrt(np.mean(rel_errors**2)) * 100.0)

    @cached_property
    def max_violation_mm(self) -> float:
        if not self.violations:
            return 0.0
        return float(max(abs(v.actual - v.expected) for v in self.violations) * 1000.0)

    @cached_property
    def per_object_rmse_mm(self) -> dict[int, float]:
        by_obj: dict[int, list[float]] = {}
        for v in self.violations:
            for oid in set((v.object_id_a, v.object_id_b)):
                by_obj.setdefault(oid, []).append(v.actual - v.expected)
        return {oid: float(np.sqrt(np.mean(np.array(errs) ** 2)) * 1000.0) for oid, errs in by_obj.items()}

    @cached_property
    def per_object_relative_rmse_pct(self) -> dict[int, float]:
        by_obj: dict[int, list[float]] = {}
        for v in self.violations:
            rel = (v.actual - v.expected) / v.expected if v.expected != 0 else 0.0
            for oid in set((v.object_id_a, v.object_id_b)):
                by_obj.setdefault(oid, []).append(rel)
        return {oid: float(np.sqrt(np.mean(np.array(errs) ** 2)) * 100.0) for oid, errs in by_obj.items()}
