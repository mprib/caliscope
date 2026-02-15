"""Scale accuracy computation for calibration quality assessment.

Computes how well the reconstructed 3D world matches known ground truth
by comparing inter-point distances between triangulated positions and
the charuco board's defined geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class FrameScaleError:
    """Per-frame scale accuracy comparing triangulated to ground truth distances.

    Sign convention: positive error = measured distance is larger than true distance
    (reconstruction is systematically too large). Computed as (measured - true).

    All distance metrics are in millimeters.
    """

    sync_index: int
    distance_rmse_mm: float
    distance_mean_signed_error_mm: float  # mean(measured - true) * 1000; separates bias from noise
    distance_max_error_mm: float  # max abs error; detects individual bad triangulations
    n_corners: int
    n_distance_pairs: int
    n_cameras_contributing: int  # cameras that observed corners at this frame
    sum_squared_errors_m2: float  # sum of (measured - true)^2 in meters^2; enables correct pooled RMSE
    centroid: tuple[float, float, float]  # mean of triangulated world points (post-alignment), NOT obj_loc


@dataclass(frozen=True)
class VolumetricScaleReport:
    """Aggregate scale accuracy across multiple frames.

    Provides pooled error metrics and per-frame breakdowns. An empty report
    (no frame_errors) is normal when no valid frames exist (e.g., pre-alignment).
    """

    frame_errors: tuple[FrameScaleError, ...]  # immutable sequence

    @cached_property
    def pooled_rmse_mm(self) -> float:
        """True pooled RMSE across all frames: sqrt(total_sse / total_pairs) * 1000."""
        if not self.frame_errors:
            return 0.0

        total_sse = sum(fe.sum_squared_errors_m2 for fe in self.frame_errors)
        total_pairs = sum(fe.n_distance_pairs for fe in self.frame_errors)

        if total_pairs == 0:
            return 0.0

        return float(np.sqrt(total_sse / total_pairs) * 1000)

    @cached_property
    def median_rmse_mm(self) -> float:
        """Median of per-frame distance RMSEs."""
        if not self.frame_errors:
            return 0.0
        return float(np.median([fe.distance_rmse_mm for fe in self.frame_errors]))

    @cached_property
    def max_rmse_mm(self) -> float:
        """Maximum per-frame RMSE."""
        if not self.frame_errors:
            return 0.0
        return float(max(fe.distance_rmse_mm for fe in self.frame_errors))

    @cached_property
    def worst_frame(self) -> FrameScaleError | None:
        """Frame with highest distance RMSE, or None if no frames."""
        if not self.frame_errors:
            return None
        return max(self.frame_errors, key=lambda fe: fe.distance_rmse_mm)

    @cached_property
    def n_frames_sampled(self) -> int:
        """Number of frames included in the report."""
        return len(self.frame_errors)

    @cached_property
    def mean_signed_error_mm(self) -> float:
        """Global bias indicator. Positive = reconstruction systematically too large.

        Weighted by n_distance_pairs per frame.
        """
        if not self.frame_errors:
            return 0.0

        weighted_sum = sum(fe.distance_mean_signed_error_mm * fe.n_distance_pairs for fe in self.frame_errors)
        total_pairs = sum(fe.n_distance_pairs for fe in self.frame_errors)

        if total_pairs == 0:
            return 0.0

        return float(weighted_sum / total_pairs)

    @cached_property
    def min_sync_index(self) -> int:
        """Minimum sync_index in the report, for sparkline frame_range."""
        if not self.frame_errors:
            return 0
        return min(fe.sync_index for fe in self.frame_errors)

    @cached_property
    def max_sync_index(self) -> int:
        """Maximum sync_index in the report, for sparkline frame_range."""
        if not self.frame_errors:
            return 0
        return max(fe.sync_index for fe in self.frame_errors)

    @classmethod
    def empty(cls) -> VolumetricScaleReport:
        """Return empty report when no valid frames exist.

        Not an error — normal pre-alignment state.
        """
        return cls(frame_errors=())


def compute_frame_scale_error(
    world_points: np.ndarray,
    object_points: np.ndarray,
    sync_index: int,
    n_cameras_contributing: int,
) -> FrameScaleError:
    """Compare triangulated inter-point distances to known ground truth at a single frame.

    Uses ALL pairwise distances, not just adjacent corners. This tests
    accuracy across the full range of measurement distances and provides
    more statistical power than adjacent-only comparisons.

    Centroid is computed as the mean of world_points, representing the
    board's 3D position in the calibrated coordinate system at this frame.

    Args:
        world_points: (N, 3) triangulated positions at reference frame (meters)
        object_points: (N, 3) ideal charuco positions (meters)
        sync_index: Frame index
        n_cameras_contributing: Number of cameras that observed corners at this frame

    Returns:
        FrameScaleError with distance error statistics

    Raises:
        ValueError: If arrays have mismatched shapes or fewer than 2 points
    """
    if world_points.shape != object_points.shape:
        raise ValueError(f"Shape mismatch: world_points {world_points.shape} vs object_points {object_points.shape}")

    n_points = len(world_points)
    if n_points < 2:
        raise ValueError(f"Need at least 2 points to compute distances, got {n_points}")

    # Compute centroid of triangulated points (board position in world space)
    mean_pos = np.mean(world_points, axis=0)
    centroid = (float(mean_pos[0]), float(mean_pos[1]), float(mean_pos[2]))

    # Compute all pairwise distances: N*(N-1)/2 pairs
    measured_distances = pdist(world_points)
    true_distances = pdist(object_points)

    # Distance errors in meters (same units as input)
    distance_errors = measured_distances - true_distances
    abs_errors = np.abs(distance_errors)

    # Compute metrics
    rmse_mm = float(np.sqrt(np.mean(distance_errors**2))) * 1000
    mean_signed_error_mm = float(np.mean(distance_errors)) * 1000  # signed, not abs
    max_error_mm = float(np.max(abs_errors)) * 1000
    sum_squared_errors_m2 = float(np.sum(distance_errors**2))  # in meters^2

    return FrameScaleError(
        sync_index=sync_index,
        distance_rmse_mm=rmse_mm,
        distance_mean_signed_error_mm=mean_signed_error_mm,
        distance_max_error_mm=max_error_mm,
        n_corners=n_points,
        n_distance_pairs=len(distance_errors),
        n_cameras_contributing=n_cameras_contributing,
        sum_squared_errors_m2=sum_squared_errors_m2,
        centroid=centroid,
    )
