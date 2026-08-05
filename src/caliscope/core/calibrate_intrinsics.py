"""Pure functions for intrinsic camera calibration.

This module provides stateless functions for calibrating camera intrinsic
parameters (camera matrix and distortion coefficients) from charuco corner
observations. These functions have no side effects and return immutable
results.

The design follows "Level 1 purity": pure functions that return results,
with mutation handled by the caller (typically a Presenter or Controller).

Main entry point: `run_intrinsic_calibration()` orchestrates the complete
workflow and returns an `IntrinsicCalibrationOutput` containing both the
calibrated camera data and a quality report.
"""

from dataclasses import dataclass, replace
import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraData
from caliscope.core.frame_selector import IntrinsicCoverageReport, select_calibration_frames
from caliscope.core.point_data import ImagePoints

logger = logging.getLogger(__name__)

# Minimum corners required per frame for OpenCV calibration
MIN_CORNERS_PER_FRAME = 4
_MIN_ROBUST_CALIBRATION_FRAMES = 8
_MIN_ROBUST_KEEP_FRACTION = 0.6
_ROBUST_SIGMA_MULTIPLIER = 3.5
_MIN_POINT_OUTLIER_THRESHOLD_PX = 1.0
_MIN_FRAME_OUTLIER_THRESHOLD_PX = 1.5


@dataclass(frozen=True)
class _CalibrationSolve:
    error: float
    camera_matrix: NDArray[np.float64]
    distortions: NDArray[np.float64]
    rvecs: tuple[NDArray, ...]
    tvecs: tuple[NDArray, ...]


@dataclass(frozen=True)
class IntrinsicCalibrationResult:
    """Immutable result of intrinsic camera calibration.

    Attributes:
        camera_matrix: 3x3 camera intrinsic matrix containing focal lengths
            and principal point coordinates.
        distortions: Distortion coefficients. Shape (5,) for standard model
            (k1, k2, p1, p2, k3) or (4,) for fisheye model (k1, k2, k3, k4).
        reprojection_error: Root mean squared reprojection error in pixels,
            as returned by cv2.calibrateCamera.
        frames_used: Number of frames used in calibration.
    """

    camera_matrix: NDArray[np.float64]
    distortions: NDArray[np.float64]
    reprojection_error: float
    frames_used: int
    used_frames: tuple[int, ...]
    rejected_frames: tuple[int, ...] = ()
    rejected_observations: int = 0


@dataclass(frozen=True)
class IntrinsicCalibrationReport:
    """Complete record of how intrinsic calibration was derived.

    Captures quality metrics, selection statistics, and provenance
    information for diagnostics and overlay restoration.
    """

    # Quality metrics
    rmse: float  # Reprojection RMSE on calibration frames (pixels)
    frames_used: int  # Number of frames used

    # Selection quality (from IntrinsicCoverageReport)
    coverage_fraction: float  # Fraction of 5x5 grid cells covered (target > 0.80)
    edge_coverage_fraction: float  # Fraction of edge cells covered (target > 0.75)
    corner_coverage_fraction: float  # Fraction of corner cells covered (target > 0.50)
    orientation_sufficient: bool  # True if >= 4 distinct tilt directions
    orientation_count: int  # Number of orientation bins covered (0-8)

    # Provenance (the ~30 selected sync_index values)
    selected_frames: tuple[int, ...]
    rejected_frames: tuple[int, ...] = ()
    rejected_observations: int = 0


@dataclass(frozen=True)
class IntrinsicCalibrationOutput:
    """Complete output of the intrinsic calibration use case.

    Bundles the calibrated camera with its quality report so they travel
    together through the system. The Coordinator persists both: camera
    parameters to camera_array.toml, report to intrinsic/reports/cam_N.toml.
    """

    camera: CameraData
    report: IntrinsicCalibrationReport


def calibrate_intrinsics(
    image_points: ImagePoints,
    cam_id: int,
    image_size: tuple[int, int],
    selected_frames: list[int],
    *,
    fisheye: bool = False,
) -> IntrinsicCalibrationResult:
    """Calibrate camera intrinsic parameters from charuco corner observations.

    This is a pure function that returns calibration results without mutating
    any input data. The caller is responsible for applying results to CameraData.

    Args:
        image_points: Detected charuco corners across all frames. Must contain
            columns: sync_index, cam_id, object_id, keypoint_id, img_loc_x, img_loc_y,
            obj_loc_x, obj_loc_y, obj_loc_z.
        cam_id: Camera cam_id to calibrate.
        image_size: (width, height) of camera images in pixels.
        selected_frames: List of sync_index values to use for calibration.
            Use select_calibration_frames() to choose optimal frames.
        fisheye: If True, use fisheye camera model (4 distortion coefficients).
            If False, use standard camera model (5 distortion coefficients).

    Returns:
        IntrinsicCalibrationResult with camera matrix, distortion coefficients,
        reprojection RMSE, and frame count.

    Raises:
        ValueError: If no valid frames found for the specified cam_id,
            or if all frames have insufficient corners (< 4 per frame).
    """
    obj_points_list, img_points_list, frame_indices = _extract_calibration_arrays(
        image_points,
        cam_id,
        selected_frames,
    )

    if len(obj_points_list) == 0:
        raise ValueError(
            f"No valid calibration frames found for cam_id {cam_id}. "
            f"Ensure frames have at least {MIN_CORNERS_PER_FRAME} corners each."
        )

    solve = _solve_intrinsics(obj_points_list, img_points_list, image_size, fisheye)
    original_frame_indices = tuple(frame_indices)
    original_observation_count = sum(len(points) for points in img_points_list)

    filtered = _filter_reprojection_outliers(
        obj_points_list,
        img_points_list,
        frame_indices,
        solve,
        fisheye,
    )
    if filtered is not None:
        filtered_obj, filtered_img, filtered_frames = filtered
        try:
            robust_solve = _solve_intrinsics(filtered_obj, filtered_img, image_size, fisheye)
        except cv2.error as exc:
            logger.warning("Intrinsic robust recalibration failed; retaining initial solve: %s", exc)
            robust_solve = None
        if robust_solve is not None and np.isfinite(robust_solve.error) and robust_solve.error <= solve.error:
            obj_points_list = filtered_obj
            img_points_list = filtered_img
            frame_indices = filtered_frames
            solve = robust_solve

    used_frame_set = set(frame_indices)
    rejected_frames = tuple(frame for frame in original_frame_indices if frame not in used_frame_set)
    rejected_observations = original_observation_count - sum(len(points) for points in img_points_list)

    logger.info(
        "Calibration complete for cam_id %s: error=%.4fpx, frames=%d, "
        "rejected_frames=%d, rejected_observations=%d",
        cam_id,
        solve.error,
        len(obj_points_list),
        len(rejected_frames),
        rejected_observations,
    )

    return IntrinsicCalibrationResult(
        camera_matrix=solve.camera_matrix,
        distortions=solve.distortions,
        reprojection_error=solve.error,
        frames_used=len(obj_points_list),
        used_frames=tuple(frame_indices),
        rejected_frames=rejected_frames,
        rejected_observations=rejected_observations,
    )


def _solve_intrinsics(
    obj_points_list: list[NDArray],
    img_points_list: list[NDArray],
    image_size: tuple[int, int],
    fisheye: bool,
) -> _CalibrationSolve:
    """Run one OpenCV intrinsic solve and retain poses for diagnostics."""
    width, height = image_size

    if fisheye:
        obj_pts = [points.reshape(-1, 1, 3).astype(np.float32) for points in obj_points_list]
        img_pts = [points.reshape(-1, 1, 2).astype(np.float32) for points in img_points_list]
        camera_matrix = np.zeros((3, 3), dtype=np.float64)
        dist_coeffs = np.zeros(4, dtype=np.float64)
        error, matrix, distortions, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_pts,
            img_pts,
            (width, height),
            camera_matrix,
            dist_coeffs,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
        )
    else:
        obj_pts = [points.astype(np.float32) for points in obj_points_list]
        img_pts = [points.astype(np.float32) for points in img_points_list]
        camera_matrix = np.eye(3, dtype=np.float64)
        camera_matrix[0, 0] = max(width, height)
        camera_matrix[1, 1] = max(width, height)
        camera_matrix[0, 2] = (width - 1) * 0.5
        camera_matrix[1, 2] = (height - 1) * 0.5
        dist_coeffs = np.zeros(5, dtype=np.float64)
        error, matrix, distortions, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts,
            img_pts,
            (width, height),
            camera_matrix,
            dist_coeffs,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        )

    return _CalibrationSolve(
        error=float(error),
        camera_matrix=np.asarray(matrix, dtype=np.float64),
        distortions=np.asarray(distortions, dtype=np.float64).ravel(),
        rvecs=tuple(np.asarray(vector) for vector in rvecs),
        tvecs=tuple(np.asarray(vector) for vector in tvecs),
    )


def _filter_reprojection_outliers(
    obj_points_list: list[NDArray],
    img_points_list: list[NDArray],
    frame_indices: list[int],
    solve: _CalibrationSolve,
    fisheye: bool,
) -> tuple[list[NDArray], list[NDArray], list[int]] | None:
    """Build a robust second-pass dataset from first-pass reprojection errors."""
    if len(frame_indices) < _MIN_ROBUST_CALIBRATION_FRAMES:
        return None

    residuals_by_frame: list[NDArray[np.float64]] = []
    for obj_points, img_points, rvec, tvec in zip(
        obj_points_list,
        img_points_list,
        solve.rvecs,
        solve.tvecs,
        strict=True,
    ):
        if fisheye:
            projected, _ = cv2.fisheye.projectPoints(
                obj_points.reshape(-1, 1, 3).astype(np.float64),
                rvec,
                tvec,
                solve.camera_matrix,
                solve.distortions.reshape(-1, 1),
            )
        else:
            projected, _ = cv2.projectPoints(
                obj_points.astype(np.float64),
                rvec,
                tvec,
                solve.camera_matrix,
                solve.distortions,
            )
        delta = projected.reshape(-1, 2) - img_points.reshape(-1, 2)
        residuals_by_frame.append(np.linalg.norm(delta, axis=1))

    all_residuals = np.concatenate(residuals_by_frame)
    if not np.isfinite(all_residuals).all():
        return None

    point_threshold = _robust_upper_threshold(
        all_residuals,
        floor=_MIN_POINT_OUTLIER_THRESHOLD_PX,
    )
    frame_rms = np.array(
        [float(np.sqrt(np.mean(residuals**2))) for residuals in residuals_by_frame],
        dtype=np.float64,
    )
    frame_threshold = _robust_upper_threshold(
        frame_rms,
        floor=_MIN_FRAME_OUTLIER_THRESHOLD_PX,
    )

    filtered_obj: list[NDArray] = []
    filtered_img: list[NDArray] = []
    filtered_frames: list[int] = []
    changed = False
    for frame, obj_points, img_points, residuals, rms in zip(
        frame_indices,
        obj_points_list,
        img_points_list,
        residuals_by_frame,
        frame_rms,
        strict=True,
    ):
        if rms > frame_threshold:
            changed = True
            continue

        keep = residuals <= point_threshold
        if int(keep.sum()) < MIN_CORNERS_PER_FRAME:
            changed = True
            continue
        if not np.all(keep):
            changed = True
        filtered_obj.append(obj_points[keep])
        filtered_img.append(img_points[keep])
        filtered_frames.append(frame)

    if not changed:
        return None

    kept_frame_fraction = len(filtered_frames) / len(frame_indices)
    original_observations = sum(len(points) for points in img_points_list)
    kept_observation_fraction = sum(len(points) for points in filtered_img) / original_observations
    if (
        len(filtered_frames) < _MIN_ROBUST_CALIBRATION_FRAMES
        or kept_frame_fraction < _MIN_ROBUST_KEEP_FRACTION
        or kept_observation_fraction < _MIN_ROBUST_KEEP_FRACTION
    ):
        logger.warning(
            "Skipping intrinsic outlier rejection because it would retain only "
            "%d/%d frames and %.0f%% of observations",
            len(filtered_frames),
            len(frame_indices),
            kept_observation_fraction * 100,
        )
        return None

    logger.info(
        "Intrinsic robust pass: point threshold %.3fpx, frame threshold %.3fpx, "
        "retained %d/%d frames",
        point_threshold,
        frame_threshold,
        len(filtered_frames),
        len(frame_indices),
    )
    return filtered_obj, filtered_img, filtered_frames


def _robust_upper_threshold(values: NDArray[np.float64], *, floor: float) -> float:
    """Median/MAD upper bound with a practical pixel-domain floor."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    return max(floor, median + _ROBUST_SIGMA_MULTIPLIER * robust_sigma)


def _extract_calibration_arrays(
    image_points: ImagePoints,
    cam_id: int,
    frames: list[int],
) -> tuple[list[NDArray], list[NDArray], list[int]]:
    """Extract per-frame object and image point arrays for OpenCV calibration.

    Returns:
        (object_points, image_points) where each is a list of (N, D) arrays
        per frame. D=3 for object points, D=2 for image points.

        Filters out frames with < MIN_CORNERS_PER_FRAME corners.
    """
    df = image_points.df

    # Filter to specified cam_id and frames
    mask = (df["cam_id"] == cam_id) & (df["sync_index"].isin(frames))
    cam_df = df[mask]

    obj_points_list: list[NDArray] = []
    img_points_list: list[NDArray] = []
    frame_indices: list[int] = []

    for sync_index in frames:
        frame_df = cam_df[cam_df["sync_index"] == sync_index]

        if len(frame_df) < MIN_CORNERS_PER_FRAME:
            logger.debug(f"Skipping frame {sync_index}: only {len(frame_df)} corners (need {MIN_CORNERS_PER_FRAME})")
            continue

        # Extract image coordinates as numpy array
        img_loc: NDArray = np.asarray(frame_df[["img_loc_x", "img_loc_y"]])

        # Extract object coordinates (3D board coordinates)
        # For planar charuco, z is typically 0 or NaN
        obj_loc: NDArray = np.asarray(frame_df[["obj_loc_x", "obj_loc_y", "obj_loc_z"]])

        # Handle NaN in obj_loc_z (planar board, z=0)
        obj_loc = np.nan_to_num(obj_loc, nan=0.0)

        obj_points_list.append(obj_loc)
        img_points_list.append(img_loc)
        frame_indices.append(sync_index)

    return obj_points_list, img_points_list, frame_indices


# =============================================================================
# Orchestrator: Main entry point
# =============================================================================


def run_intrinsic_calibration(
    camera: CameraData,
    image_points: ImagePoints,
    selection_result: IntrinsicCoverageReport | None = None,
) -> IntrinsicCalibrationOutput:
    """Execute complete intrinsic calibration workflow.

    1. Frame selection (if not provided)
    2. Intrinsic calibration -> matrix, distortions, rmse
    3. Build calibrated CameraData
    4. Build IntrinsicCalibrationReport
    5. Return both together

    Args:
        camera: Camera to calibrate (provides cam_id, size, fisheye flag).
        image_points: Detected charuco corners across all frames.
        selection_result: Pre-computed frame selection. If None, runs
            `select_calibration_frames()` automatically.

    Returns:
        IntrinsicCalibrationOutput with calibrated camera and quality report.

    Raises:
        ValueError: If no valid frames found or calibration fails.
    """
    cam_id = camera.cam_id
    image_size = camera.size
    fisheye = camera.fisheye

    # Step 1: Frame selection (if not provided)
    if selection_result is None:
        selection_result = select_calibration_frames(image_points, cam_id, image_size)

    if not selection_result.selected_frames:
        raise ValueError(f"No frames selected for calibration on cam_id {cam_id}")

    selected_frames = selection_result.selected_frames

    # Step 2: Intrinsic calibration
    calibration_result = calibrate_intrinsics(
        image_points,
        cam_id,
        image_size,
        selected_frames,
        fisheye=fisheye,
    )

    # Step 3: Build calibrated CameraData
    calibrated_camera = replace(
        camera,
        matrix=calibration_result.camera_matrix,
        distortions=calibration_result.distortions,
        error=calibration_result.reprojection_error,
        grid_count=calibration_result.frames_used,
    )

    # Step 4: Build report
    report = IntrinsicCalibrationReport(
        rmse=calibration_result.reprojection_error,
        frames_used=calibration_result.frames_used,
        coverage_fraction=selection_result.coverage_fraction,
        edge_coverage_fraction=selection_result.edge_coverage_fraction,
        corner_coverage_fraction=selection_result.corner_coverage_fraction,
        orientation_sufficient=selection_result.orientation_sufficient,
        orientation_count=selection_result.orientation_count,
        selected_frames=calibration_result.used_frames,
        rejected_frames=calibration_result.rejected_frames,
        rejected_observations=calibration_result.rejected_observations,
    )

    logger.info(
        f"Calibration complete for cam_id {cam_id}: "
        f"rmse={report.rmse:.3f}px, "
        f"frames={report.frames_used}, "
        f"coverage={report.coverage_fraction:.0%}"
    )

    return IntrinsicCalibrationOutput(camera=calibrated_camera, report=report)
