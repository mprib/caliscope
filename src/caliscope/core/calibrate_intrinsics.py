"""Pure functions for intrinsic camera calibration.

This module provides stateless functions for calibrating camera intrinsic
parameters (camera matrix and distortion coefficients) from charuco corner
observations. Unlike the legacy IntrinsicCalibrator class, these functions
have no side effects and return immutable results.

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


@dataclass(frozen=True)
class IntrinsicCalibrationOutput:
    """Complete output of the intrinsic calibration use case.

    Bundles the calibrated camera with its quality report so they travel
    together through the system. The Coordinator persists both: camera
    parameters to camera_array.toml, report to intrinsic/reports/port_N.toml.
    """

    camera: CameraData
    report: IntrinsicCalibrationReport


def calibrate_intrinsics(
    image_points: ImagePoints,
    port: int,
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
            columns: sync_index, port, point_id, img_loc_x, img_loc_y,
            obj_loc_x, obj_loc_y, obj_loc_z.
        port: Camera port to calibrate.
        image_size: (width, height) of camera images in pixels.
        selected_frames: List of sync_index values to use for calibration.
            Use select_calibration_frames() to choose optimal frames.
        fisheye: If True, use fisheye camera model (4 distortion coefficients).
            If False, use standard camera model (5 distortion coefficients).

    Returns:
        IntrinsicCalibrationResult with camera matrix, distortion coefficients,
        reprojection RMSE, and frame count.

    Raises:
        ValueError: If no valid frames found for the specified port,
            or if all frames have insufficient corners (< 4 per frame).
    """
    obj_points_list, img_points_list = _extract_calibration_arrays(image_points, port, selected_frames)

    if len(obj_points_list) == 0:
        raise ValueError(
            f"No valid calibration frames found for port {port}. "
            f"Ensure frames have at least {MIN_CORNERS_PER_FRAME} corners each."
        )

    width, height = image_size

    if fisheye:
        # Fisheye requires specific array shapes: (N, 1, D)
        obj_pts = [p.reshape(-1, 1, 3).astype(np.float32) for p in obj_points_list]
        img_pts = [p.reshape(-1, 1, 2).astype(np.float32) for p in img_points_list]

        # Pre-initialize output matrices (required for fisheye.calibrate)
        camera_matrix = np.zeros((3, 3), dtype=np.float64)
        dist_coeffs = np.zeros(4, dtype=np.float64)

        error, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_pts,
            img_pts,
            (width, height),
            camera_matrix,
            dist_coeffs,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
        )
        # fisheye.calibrate returns dist as (4,) already
        dist = dist.ravel()
    else:
        # Standard calibration accepts (N, D) arrays
        obj_pts = [p.astype(np.float32) for p in obj_points_list]
        img_pts = [p.astype(np.float32) for p in img_points_list]

        # Pre-initialize output matrices (required by type checker, works at runtime)
        camera_matrix = np.zeros((3, 3), dtype=np.float64)
        dist_coeffs = np.zeros(5, dtype=np.float64)

        error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts,
            img_pts,
            (width, height),
            camera_matrix,
            dist_coeffs,
        )
        # calibrateCamera returns dist as (1, 5), flatten it
        dist = dist.ravel()

    logger.info(f"Calibration complete for port {port}: error={error:.4f}px, frames={len(obj_points_list)}")

    return IntrinsicCalibrationResult(
        camera_matrix=np.asarray(mtx, dtype=np.float64),
        distortions=np.asarray(dist, dtype=np.float64),
        reprojection_error=float(error),
        frames_used=len(obj_points_list),
    )


def _extract_calibration_arrays(
    image_points: ImagePoints,
    port: int,
    frames: list[int],
) -> tuple[list[NDArray], list[NDArray]]:
    """Extract per-frame object and image point arrays for OpenCV calibration.

    Returns:
        (object_points, image_points) where each is a list of (N, D) arrays
        per frame. D=3 for object points, D=2 for image points.

        Filters out frames with < MIN_CORNERS_PER_FRAME corners.
    """
    df = image_points.df

    # Filter to specified port and frames
    mask = (df["port"] == port) & (df["sync_index"].isin(frames))
    port_df = df[mask]

    obj_points_list: list[NDArray] = []
    img_points_list: list[NDArray] = []

    for sync_index in frames:
        frame_df = port_df[port_df["sync_index"] == sync_index]

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

    return obj_points_list, img_points_list


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
        camera: Camera to calibrate (provides port, size, fisheye flag).
        image_points: Detected charuco corners across all frames.
        selection_result: Pre-computed frame selection. If None, runs
            `select_calibration_frames()` automatically.

    Returns:
        IntrinsicCalibrationOutput with calibrated camera and quality report.

    Raises:
        ValueError: If no valid frames found or calibration fails.
    """
    port = camera.port
    image_size = camera.size
    fisheye = camera.fisheye

    # Step 1: Frame selection (if not provided)
    if selection_result is None:
        selection_result = select_calibration_frames(image_points, port, image_size)

    if not selection_result.selected_frames:
        raise ValueError(f"No frames selected for calibration on port {port}")

    selected_frames = selection_result.selected_frames

    # Step 2: Intrinsic calibration
    calibration_result = calibrate_intrinsics(
        image_points,
        port,
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
        selected_frames=tuple(selected_frames),
    )

    logger.info(
        f"Calibration complete for port {port}: "
        f"rmse={report.rmse:.3f}px, "
        f"frames={report.frames_used}, "
        f"coverage={report.coverage_fraction:.0%}"
    )

    return IntrinsicCalibrationOutput(camera=calibrated_camera, report=report)
