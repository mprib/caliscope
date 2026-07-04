"""Scale accuracy computation for calibration quality assessment.

Computes how well the reconstructed 3D world matches known ground truth
by comparing inter-point distances between triangulated positions and
the charuco board's defined geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from caliscope.core.capture_volume import CaptureVolume

import numpy as np
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class FrameScaleError:
    """Per-frame, per-object scale accuracy comparing triangulated to ground truth distances.

    Sign convention: positive error = measured distance is larger than true distance
    (reconstruction is systematically too large). Computed as (measured - true).

    All distance metrics are in millimeters.
    """

    sync_index: int
    object_id: int
    distance_rmse_mm: float
    distance_mean_signed_error_mm: float  # mean(measured - true) * 1000; separates bias from noise
    distance_max_error_mm: float  # max abs error; detects individual bad triangulations
    n_corners: int
    n_distance_pairs: int
    n_cameras_contributing: int  # cameras that observed corners at this frame
    sum_squared_errors_m2: float  # sum of (measured - true)^2 in meters^2; enables correct pooled RMSE
    sum_squared_relative_errors: float  # sum of ((measured - true) / D_ref)^2; D_ref is per-object nominal diagonal
    centroid: tuple[float, float, float]  # mean of triangulated world points (post-alignment), NOT obj_loc


@dataclass(frozen=True)
class VolumetricScaleReport:
    """Aggregate scale accuracy across multiple frames.

    Provides pooled error metrics and per-frame breakdowns. An empty report
    (no frame_errors) is normal when no valid frames exist (e.g., pre-alignment).
    """

    frame_errors: tuple[FrameScaleError, ...]  # immutable sequence
    static_object_ids: frozenset[int] = frozenset()

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

    @cached_property
    def pooled_relative_rmse_pct(self) -> float:
        """True pooled relative RMSE across all frames: sqrt(total_sse_rel / total_pairs) * 100."""
        if not self.frame_errors:
            return 0.0

        total_sse_rel = sum(fe.sum_squared_relative_errors for fe in self.frame_errors)
        total_pairs = sum(fe.n_distance_pairs for fe in self.frame_errors)

        if total_pairs == 0:
            return 0.0

        return float(np.sqrt(total_sse_rel / total_pairs) * 100)

    @cached_property
    def per_frame_relative_rmse_pct(self) -> dict[int, float]:
        """Pooled relative RMSE % grouped by sync_index. Excludes static markers (STATIC_SYNC_INDEX)."""
        from caliscope.core.point_data import STATIC_SYNC_INDEX

        by_frame: dict[int, tuple[float, int]] = {}
        for fe in self.frame_errors:
            if fe.sync_index == STATIC_SYNC_INDEX:
                continue
            sse, pairs = by_frame.get(fe.sync_index, (0.0, 0))
            by_frame[fe.sync_index] = (sse + fe.sum_squared_relative_errors, pairs + fe.n_distance_pairs)

        return {si: float(np.sqrt(sse / pairs) * 100) for si, (sse, pairs) in by_frame.items() if pairs > 0}

    @cached_property
    def per_object_relative_rmse_pct(self) -> dict[int, float]:
        """Pooled relative RMSE % grouped by object_id. Includes static markers."""
        by_object: dict[int, tuple[float, int]] = {}
        for fe in self.frame_errors:
            sse, pairs = by_object.get(fe.object_id, (0.0, 0))
            by_object[fe.object_id] = (sse + fe.sum_squared_relative_errors, pairs + fe.n_distance_pairs)

        return {oid: float(np.sqrt(sse / pairs) * 100) for oid, (sse, pairs) in by_object.items() if pairs > 0}

    @cached_property
    def split_relative_rmse_pct(self) -> tuple[float | None, float | None]:
        """Pooled relative RMSE % split into (moving, static) using static_object_ids.

        A side with no contributing pairs returns None.
        """
        moving_sse = 0.0
        moving_pairs = 0
        static_sse = 0.0
        static_pairs = 0

        for fe in self.frame_errors:
            if fe.object_id in self.static_object_ids:
                static_sse += fe.sum_squared_relative_errors
                static_pairs += fe.n_distance_pairs
            else:
                moving_sse += fe.sum_squared_relative_errors
                moving_pairs += fe.n_distance_pairs

        moving_pct = float(np.sqrt(moving_sse / moving_pairs) * 100) if moving_pairs > 0 else None
        static_pct = float(np.sqrt(static_sse / static_pairs) * 100) if static_pairs > 0 else None
        return (moving_pct, static_pct)

    @classmethod
    def empty(cls) -> VolumetricScaleReport:
        """Return empty report when no valid frames exist.

        Not an error — normal pre-alignment state.
        """
        return cls(frame_errors=())


def compute_depth_ratios(capture_volume: "CaptureVolume") -> dict[int, float]:
    """Per cam_id: p95(z)/p5(z) of moving world points in that camera's frame.

    Excludes STATIC_SYNC_INDEX rows and non-positive depths. Cameras with < 2
    valid depths map to float('nan').
    """
    from caliscope.core.point_data import STATIC_SYNC_INDEX

    world_df = capture_volume.world_points.df
    moving = world_df[world_df["sync_index"] != STATIC_SYNC_INDEX]
    if moving.empty:
        return {cam_id: float("nan") for cam_id in capture_volume.camera_array.posed_cameras}

    pts = moving[["x_coord", "y_coord", "z_coord"]].to_numpy()

    ratios: dict[int, float] = {}
    for cam_id, cam in capture_volume.camera_array.posed_cameras.items():
        assert cam.rotation is not None and cam.translation is not None
        z = (cam.rotation @ pts.T).T[:, 2] + cam.translation[2]
        z = z[z > 0]
        if len(z) < 2:
            ratios[cam_id] = float("nan")
        else:
            ratios[cam_id] = float(np.percentile(z, 95) / np.percentile(z, 5))
    return ratios


def compute_frame_scale_error(
    world_points: np.ndarray,
    object_points: np.ndarray,
    sync_index: int,
    object_id: int,
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

    # Per-object normalization: one D_ref (nominal diagonal) shared by all pairs of this object.
    # Per-pair normalization would inflate short pairs, since triangulation error is ~constant in mm.
    D_ref = float(np.max(true_distances))
    sum_squared_relative_errors = sum_squared_errors_m2 / (D_ref**2) if D_ref > 0 else 0.0

    return FrameScaleError(
        sync_index=sync_index,
        object_id=object_id,
        distance_rmse_mm=rmse_mm,
        distance_mean_signed_error_mm=mean_signed_error_mm,
        distance_max_error_mm=max_error_mm,
        n_corners=n_points,
        n_distance_pairs=len(distance_errors),
        n_cameras_contributing=n_cameras_contributing,
        sum_squared_errors_m2=sum_squared_errors_m2,
        sum_squared_relative_errors=sum_squared_relative_errors,
        centroid=centroid,
    )
