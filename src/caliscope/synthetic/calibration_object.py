"""Calibration object with known point geometry in local frame."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationObject:
    """Rigid body with known point geometry in local frame.

    Foundation is an arbitrary (N, 3) point cloud, enabling:
    - Planar grids (charuco-like calibration boards)
    - Non-planar objects (future: markerless calibration)
    - Canonical face/body models (future: PnP-based bootstrapping)

    Points are in object-local coordinates (mm). When combined with a
    trajectory, these are transformed to world coordinates per frame.

    Attributes:
        points: (N, 3) points in object-local coordinates
        point_ids: (N,) unique integer identifiers for each point
    """

    points: NDArray[np.float64]
    point_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Validate point arrays."""
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Points must be (N, 3), got shape {self.points.shape}")
        if self.point_ids.ndim != 1:
            raise ValueError(f"Point IDs must be 1D, got shape {self.point_ids.shape}")
        if len(self.points) != len(self.point_ids):
            raise ValueError(f"Points and IDs must have same length: {len(self.points)} vs {len(self.point_ids)}")
        if len(self.point_ids) != len(np.unique(self.point_ids)):
            raise ValueError("Point IDs must be unique")
        if len(self.points) < 4:
            raise ValueError(f"Need at least 4 points for calibration, got {len(self.points)}")

    @classmethod
    def planar_grid(
        cls,
        rows: int,
        cols: int,
        spacing_mm: float,
        origin: str = "corner",
    ) -> CalibrationObject:
        """Create rectangular planar grid (charuco-like).

        Grid lies in the XY plane (Z=0) of object-local coordinates.

        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            spacing_mm: Distance between adjacent grid points
            origin: Where to place local origin
                - "corner": Origin at (0, 0), grid extends in +X and +Y
                - "center": Origin at grid centroid

        Returns:
            CalibrationObject with rows*cols points

        Raises:
            ValueError: If rows < 2 or cols < 2 or spacing <= 0
        """
        if rows < 2 or cols < 2:
            raise ValueError(f"Grid must be at least 2x2, got {rows}x{cols}")
        if spacing_mm <= 0:
            raise ValueError(f"Spacing must be positive, got {spacing_mm}")

        n_points = rows * cols
        points = np.zeros((n_points, 3), dtype=np.float64)
        point_ids = np.zeros(n_points, dtype=np.int64)

        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                points[idx, 0] = col * spacing_mm
                points[idx, 1] = row * spacing_mm
                points[idx, 2] = 0.0
                point_ids[idx] = idx

        if origin == "center":
            centroid = points.mean(axis=0)
            points = points - centroid

        return cls(points=points, point_ids=point_ids)

    @classmethod
    def from_points(
        cls,
        points: NDArray[np.float64],
        point_ids: NDArray[np.int64] | None = None,
    ) -> CalibrationObject:
        """Create from arbitrary point cloud.

        Args:
            points: (N, 3) array of 3D points
            point_ids: (N,) array of unique IDs. If None, auto-generates 0..N-1

        Returns:
            CalibrationObject with the provided points
        """
        points = np.asarray(points, dtype=np.float64)
        if point_ids is None:
            point_ids = np.arange(len(points), dtype=np.int64)
        else:
            point_ids = np.asarray(point_ids, dtype=np.int64)

        return cls(points=points, point_ids=point_ids)

    @property
    def n_points(self) -> int:
        """Number of points in the object."""
        return len(self.points)

    @cached_property
    def centroid(self) -> NDArray[np.float64]:
        """Center of mass of point cloud (for visualization centering)."""
        return self.points.mean(axis=0)

    @cached_property
    def extent(self) -> float:
        """Maximum distance from centroid to any point (bounding sphere radius)."""
        distances = np.linalg.norm(self.points - self.centroid, axis=1)
        return float(np.max(distances))
