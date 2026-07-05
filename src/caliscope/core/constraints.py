from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import rtoml

from caliscope.core.aruco_marker import ArucoMarkerSet


@dataclass(frozen=True)
class DistanceConstraint:
    object_id_a: int
    keypoint_id_a: int
    object_id_b: int
    keypoint_id_b: int
    distance: float
    sigma: float


@dataclass(frozen=True)
class ConstraintSet:
    distances: tuple[DistanceConstraint, ...]
    static_object_ids: frozenset[int]

    @classmethod
    def from_marker_set(
        cls,
        marker_set: ArucoMarkerSet,
        sigma_m: float = 0.002,
    ) -> ConstraintSet:
        """Compile distance constraints from a marker set.

        Emits 6 intra-marker constraints per marker (4 edges + 2 diagonals)
        and 4 cross-marker constraints per link. All distances in meters.
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

        for link in marker_set.links:
            for i in range(4):
                constraints.append(
                    DistanceConstraint(
                        object_id_a=link.marker_a,
                        keypoint_id_a=i,
                        object_id_b=link.marker_b,
                        keypoint_id_b=link.corner_map[i],
                        distance=link.separation_m,
                        sigma=sigma_m,
                    )
                )

        static_ids = frozenset(mid for mid, m in marker_set.markers.items() if m.static)

        return cls(distances=tuple(constraints), static_object_ids=static_ids)

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
            static_ids = frozenset(data.get("static_object_ids", []))
            return cls(distances=distances, static_object_ids=static_ids)
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
