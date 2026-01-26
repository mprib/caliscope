"""Scale accuracy computation for calibration quality assessment.

Computes how well the reconstructed 3D world matches known ground truth
by comparing inter-point distances between triangulated positions and
the charuco board's defined geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class ScaleAccuracyData:
    """Scale accuracy metrics comparing triangulated to ground truth distances.

    Uses ALL pairwise distances between detected corners, not just adjacent ones.
    For N corners, this gives N*(N-1)/2 distance comparisons, testing accuracy
    across the full range of measurement scales (adjacent ~30mm to diagonal ~280mm).
    """

    reference_sync_index: int
    n_corners_detected: int
    n_distance_pairs: int
    distance_rmse_mm: float
    distance_mean_error_mm: float
    distance_max_error_mm: float
    relative_error_percent: float

    @classmethod
    def empty(cls) -> ScaleAccuracyData:
        """Create empty data for when no reference frame is available."""
        return cls(
            reference_sync_index=-1,
            n_corners_detected=0,
            n_distance_pairs=0,
            distance_rmse_mm=0.0,
            distance_mean_error_mm=0.0,
            distance_max_error_mm=0.0,
            relative_error_percent=0.0,
        )


def compute_scale_accuracy(
    world_points: np.ndarray,
    object_points: np.ndarray,
    sync_index: int,
) -> ScaleAccuracyData:
    """Compare triangulated inter-point distances to known ground truth.

    Uses ALL pairwise distances, not just adjacent corners. This tests
    accuracy across the full range of measurement distances and provides
    more statistical power than adjacent-only comparisons.

    Args:
        world_points: (N, 3) triangulated positions at reference frame (meters)
        object_points: (N, 3) ideal charuco positions (meters)
        sync_index: Frame index used as reference

    Returns:
        ScaleAccuracyData with distance error statistics

    Raises:
        ValueError: If arrays have mismatched shapes or fewer than 2 points
    """
    if world_points.shape != object_points.shape:
        raise ValueError(f"Shape mismatch: world_points {world_points.shape} vs object_points {object_points.shape}")

    n_points = len(world_points)
    if n_points < 2:
        raise ValueError(f"Need at least 2 points to compute distances, got {n_points}")

    # Compute all pairwise distances: N*(N-1)/2 pairs
    measured_distances = pdist(world_points)
    true_distances = pdist(object_points)

    # Distance errors in same units as input (typically meters)
    distance_errors = measured_distances - true_distances
    abs_errors = np.abs(distance_errors)

    # Convert to mm for display (assuming inputs are in meters)
    rmse_mm = float(np.sqrt(np.mean(distance_errors**2))) * 1000
    mean_error_mm = float(np.mean(abs_errors)) * 1000
    max_error_mm = float(np.max(abs_errors)) * 1000

    # Relative error as percentage of mean true distance
    mean_true_distance = float(np.mean(true_distances))
    relative_error = 100.0 * (rmse_mm / 1000) / mean_true_distance if mean_true_distance > 0 else 0.0

    return ScaleAccuracyData(
        reference_sync_index=sync_index,
        n_corners_detected=n_points,
        n_distance_pairs=len(distance_errors),
        distance_rmse_mm=rmse_mm,
        distance_mean_error_mm=mean_error_mm,
        distance_max_error_mm=max_error_mm,
        relative_error_percent=relative_error,
    )
