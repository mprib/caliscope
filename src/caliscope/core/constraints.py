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
class ConstraintSet:
    distances: tuple[DistanceConstraint, ...]
    static_object_ids: frozenset[int]
    centroid_distances: tuple[CentroidDistanceConstraint, ...] = ()

    @classmethod
    def from_marker_set(
        cls,
        marker_set: ArucoMarkerSet,
        sigma_m: float = 0.002,
        center_sigma_m: float = 0.005,
    ) -> ConstraintSet:
        """Compile distance constraints from a marker set.

        Emits 6 intra-marker constraints per marker (4 edges + 2 diagonals).
        Each corner DistanceLink passes through as exactly one
        DistanceConstraint — the user's measured distance is the constraint,
        no derived pairs. Each center DistanceLink compiles to one
        CentroidDistanceConstraint. A link's own sigma_m wins when set;
        otherwise corner links default to sigma_m and center links default
        to center_sigma_m. All distances in meters.
        """
        constraints: list[DistanceConstraint] = []

        for marker_id, marker in marker_set.markers.items():
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

        static_ids = frozenset(mid for mid, m in marker_set.markers.items() if m.static)

        return cls(
            distances=tuple(constraints),
            static_object_ids=static_ids,
            centroid_distances=tuple(centroid_constraints),
        )

    @classmethod
    def from_charuco(cls, charuco: Charuco, sigma_m: float = 0.002) -> ConstraintSet:
        """Compile board-geometry distance constraints for BA.

        object_id is 0 for every corner, matching CharucoTracker's identity
        scheme (object_id=0, keypoint_id=chessboard corner index).
        static_object_ids is empty — the board moves. Produces only
        DistanceConstraints, never centroids (a charuco board has no separate
        markers to link by centroid).

        Corner ids come from `charuco.board.getChessboardCorners()`, the same
        (N, 3) array CharucoTracker indexes by `keypoint_id`. Edges are found
        by corner *coordinates*, not assumed index order — the same robust
        pattern `Charuco.get_connected_points` uses — because OpenCV does not
        guarantee any particular corner-id-to-grid-position layout.

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
        corners = np.asarray(charuco.board.getChessboardCorners())
        square_length = float(charuco.board.getSquareLength())

        # Quantize by rounding each coordinate to the nearest multiple of
        # square_length. OpenCV emits these as float32, so exact equality is
        # unsafe; unlike a fixed absolute tolerance, this scales with board
        # size and tolerates float32 error (empirically up to ~1e-6 of a
        # square) with wide margin below the 0.5-square ambiguity band.
        x_keys = np.round(corners[:, 0] / square_length).astype(np.int64)
        y_keys = np.round(corners[:, 1] / square_length).astype(np.int64)

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

        constraints = tuple(
            DistanceConstraint(
                object_id_a=0,
                keypoint_id_a=a,
                object_id_b=0,
                keypoint_id_b=b,
                distance=float(np.linalg.norm(corners[a] - corners[b])),
                sigma=sigma_m,
            )
            for a, b in edges
        )

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
            "centroid_distances": [
                {
                    "object_id_a": c.object_id_a,
                    "object_id_b": c.object_id_b,
                    "distance": c.distance,
                    "sigma": c.sigma,
                }
                for c in self.centroid_distances
            ],
        }
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
            static_ids = frozenset(data.get("static_object_ids", []))
            return cls(distances=distances, static_object_ids=static_ids, centroid_distances=centroid_distances)
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
