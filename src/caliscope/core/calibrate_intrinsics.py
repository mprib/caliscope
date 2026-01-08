"""Pure functions for intrinsic camera calibration.

This module provides stateless functions for calibrating camera intrinsic
parameters (camera matrix and distortion coefficients) from charuco corner
observations. Unlike the legacy IntrinsicCalibrator class, these functions
have no side effects and return immutable results.

The design follows "Level 1 purity": pure functions that return results,
with mutation handled by the caller (typically a Presenter or Controller).
"""

from dataclasses import dataclass
import logging

import cv2
import numpy as np
from numpy.typing import NDArray

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
class HoldoutResult:
    """Results from out-of-sample reprojection error evaluation.

    Attributes:
        rmse: Aggregate RMSE across all held-out observations in normalized
            coordinates. Multiply by mean focal length for approximate pixels.
        rmse_pixels: Approximate RMSE in pixel units (rmse * mean_focal_length).
        per_frame_rmse: Per-frame RMSE for diagnostics. Keys are sync_index.
        total_points: Total points evaluated across all successful frames.
        total_frames: Number of held-out frames attempted.
        failed_frames: sync_index values where solvePnP failed.
    """

    rmse: float
    rmse_pixels: float
    per_frame_rmse: dict[int, float]
    total_points: int
    total_frames: int
    failed_frames: list[int]


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


def compute_holdout_error(
    image_points: ImagePoints,
    result: IntrinsicCalibrationResult,
    port: int,
    holdout_frames: list[int],
    *,
    fisheye: bool = False,
) -> HoldoutResult:
    """Compute out-of-sample reprojection error on held-out frames.

    For each held-out frame:
    1. Extract detected 2D corners and corresponding 3D object points
    2. Undistort 2D points to normalized coordinates
    3. Estimate board pose using solvePnP with identity K (since we undistorted)
    4. Project 3D object points back to normalized coordinates
    5. Compute error between projected and undistorted points

    This provides a cross-validation metric for calibration quality. If the
    calibration overfits to training frames, holdout RMSE will be higher.

    Args:
        image_points: Detected charuco corners (same source as calibration).
        result: Result from calibrate_intrinsic().
        port: Camera port.
        holdout_frames: List of sync_index values NOT used in calibration.
        fisheye: Must match the value used in calibrate_intrinsic().

    Returns:
        HoldoutResult with aggregate RMSE, per-frame RMSE, and failure info.
    """
    per_frame_errors: dict[int, NDArray] = {}
    failed_frames: list[int] = []

    for sync_index in holdout_frames:
        frame_result = _evaluate_frame(
            image_points,
            result.camera_matrix,
            result.distortions,
            port,
            sync_index,
            fisheye=fisheye,
        )

        if frame_result is None:
            failed_frames.append(sync_index)
        else:
            per_frame_errors[sync_index] = frame_result

    # Compute aggregate statistics
    if per_frame_errors:
        all_errors = np.vstack(list(per_frame_errors.values()))
        rmse = float(np.sqrt(np.mean(np.sum(all_errors**2, axis=1))))
        total_points = sum(len(e) for e in per_frame_errors.values())

        per_frame_rmse = {idx: float(np.sqrt(np.mean(np.sum(err**2, axis=1)))) for idx, err in per_frame_errors.items()}

        # Approximate pixel RMSE using mean focal length
        fx, fy = result.camera_matrix[0, 0], result.camera_matrix[1, 1]
        mean_focal = (fx + fy) / 2
        rmse_pixels = rmse * mean_focal
    else:
        rmse = float("nan")
        rmse_pixels = float("nan")
        per_frame_rmse = {}
        total_points = 0

    return HoldoutResult(
        rmse=rmse,
        rmse_pixels=rmse_pixels,
        per_frame_rmse=per_frame_rmse,
        total_points=total_points,
        total_frames=len(holdout_frames),
        failed_frames=failed_frames,
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


def _undistort_points(
    points: NDArray,
    camera_matrix: NDArray,
    distortions: NDArray,
    *,
    fisheye: bool = False,
) -> NDArray:
    """Undistort 2D points to normalized coordinates.

    Args:
        points: (N, 2) array of distorted pixel coordinates.
        camera_matrix: 3x3 camera intrinsic matrix.
        distortions: Distortion coefficients.
        fisheye: Whether to use fisheye model.

    Returns:
        (N, 2) array of undistorted normalized coordinates.
    """
    points_reshaped = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 1, 2)

    # Output in normalized coordinates (identity projection matrix)
    P = np.eye(3, dtype=np.float64)

    if fisheye:
        undistorted = cv2.fisheye.undistortPoints(points_reshaped, camera_matrix, distortions, P=P)
    else:
        undistorted = cv2.undistortPoints(points_reshaped, camera_matrix, distortions, P=P)

    return undistorted.reshape(-1, 2)


def _estimate_pose_for_frame(
    img_points_normalized: NDArray,
    obj_points: NDArray,
) -> tuple[NDArray, NDArray] | None:
    """Estimate board pose from normalized 2D points and 3D object points.

    Uses solvePnP with identity K (since points are already normalized).

    Args:
        img_points_normalized: (N, 2) undistorted normalized coordinates.
        obj_points: (N, 3) 3D object coordinates on the board.

    Returns:
        (rvec, tvec) if successful, None if pose estimation failed.
    """
    if len(img_points_normalized) < MIN_CORNERS_PER_FRAME:
        return None

    # Check for degenerate point distribution
    if not _points_are_well_distributed(img_points_normalized):
        return None

    # Identity K and zero distortion since we're using normalized coordinates
    K = np.eye(3, dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)

    # Prepare arrays for OpenCV
    obj_pts = obj_points.astype(np.float32)
    img_pts = img_points_normalized.astype(np.float32)

    # Try IPPE first (optimal for planar targets like charuco boards)
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE)

    if not success:
        # Fallback to iterative solver
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        return None

    # Sanity check: verify pose produces reasonable reprojection
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    reprojection_error = np.sqrt(np.mean((img_points_normalized - projected.reshape(-1, 2)) ** 2))

    # Reject poses with unreasonably high error (in normalized units)
    # 0.1 normalized ~ 50-100 pixels depending on focal length
    MAX_ACCEPTABLE_ERROR = 0.1
    if reprojection_error > MAX_ACCEPTABLE_ERROR:
        logger.debug(f"Rejecting pose with error {reprojection_error:.4f} (threshold {MAX_ACCEPTABLE_ERROR})")
        return None

    # Verify board is in front of camera (z > 0)
    if tvec[2, 0] < 0:
        logger.debug("Rejecting pose with negative z (board behind camera)")
        return None

    return rvec, tvec


def _points_are_well_distributed(points: NDArray, min_span: float = 0.01) -> bool:
    """Check if points span sufficient area for stable pose estimation.

    Args:
        points: (N, 2) array of points.
        min_span: Minimum required span in each dimension.

    Returns:
        True if points are sufficiently distributed.
    """
    if len(points) < MIN_CORNERS_PER_FRAME:
        return False

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    span = max_coords - min_coords

    return bool(span[0] > min_span and span[1] > min_span)


def _evaluate_frame(
    image_points: ImagePoints,
    camera_matrix: NDArray,
    distortions: NDArray,
    port: int,
    sync_index: int,
    *,
    fisheye: bool = False,
) -> NDArray | None:
    """Evaluate reprojection error for a single held-out frame.

    Returns:
        (N, 2) array of errors (undistorted - projected) in normalized
        coordinates, or None if evaluation failed.
    """
    df = image_points.df
    frame_df = df[(df["port"] == port) & (df["sync_index"] == sync_index)]

    if len(frame_df) < MIN_CORNERS_PER_FRAME:
        return None

    # Extract points as numpy arrays
    img_loc: NDArray = np.asarray(frame_df[["img_loc_x", "img_loc_y"]])
    obj_loc: NDArray = np.asarray(frame_df[["obj_loc_x", "obj_loc_y", "obj_loc_z"]])
    obj_loc = np.nan_to_num(obj_loc, nan=0.0)

    # Undistort to normalized coordinates
    img_normalized = _undistort_points(img_loc, camera_matrix, distortions, fisheye=fisheye)

    # Estimate board pose
    pose = _estimate_pose_for_frame(img_normalized, obj_loc)
    if pose is None:
        return None

    rvec, tvec = pose

    # Project object points to normalized coordinates
    K = np.eye(3, dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    projected, _ = cv2.projectPoints(obj_loc.astype(np.float32), rvec, tvec, K, D)

    # Compute error
    errors = img_normalized - projected.reshape(-1, 2)
    return errors
