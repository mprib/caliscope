# caliscope/calibration/array_initialization/estimate_pairwise_extrinsics.py
from __future__ import annotations

import logging
from itertools import combinations

import cv2
import numpy as np
import pandas as pd

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bootstrap_pose.paired_pose_network import PairedPoseNetwork
from caliscope.core.bootstrap_pose.stereopairs import StereoPair
from caliscope.core.point_data import ImagePoints

logger = logging.getLogger(__name__)


def build_legacy_stereocal_paired_pose_network(
    image_points: ImagePoints,
    camera_array: CameraArray,
    boards_sampled: int = 10,
) -> PairedPoseNetwork:
    """
    Legacy stereo calibration algorithm that estimates pairwise extrinsics
    between all camera pairs in the array.

    This is a refactoring of the original StereoCalibrator class into a
    module-level function that returns a StereoPairGraph instead of a raw dictionary.

    Args:
        image_points: Validated 2D point correspondences across cameras
        camera_array: Camera array with intrinsic parameters
        boards_sampled: Number of sync indices to sample for each pair

    Returns:
        StereoPairGraph containing all successfully estimated stereo pairs
    """
    logger.info("Beginning pairwise extrinsic estimation...")

    # Get all camera cam_ids that have intrinsic calibration AND are not ignored
    cam_ids = [
        cam_id
        for cam_id, cam in camera_array.cameras.items()
        if cam.matrix is not None and cam.distortions is not None and not cam.ignore
    ]

    if len(cam_ids) < 2:
        logger.error("Need at least 2 calibrated cameras for stereo estimation")
        return PairedPoseNetwork(_pairs={})

    # Pre-compute coverage regions ONCE for all pairs
    points_with_coverage = _add_coverage_regions(image_points.df)

    # Build stereo pairs for all combinations
    pairs = {}
    for cam_id_a, cam_id_b in combinations(cam_ids, 2):
        # logger.info(f"Estimating stereo pair for cameras {cam_id_a}-{cam_id_b}")

        stereo_pair = _estimate_single_pair(
            points_with_coverage=points_with_coverage,
            camera_array=camera_array,
            cam_id_a=cam_id_a,
            cam_id_b=cam_id_b,
            boards_sampled=boards_sampled,
        )

        if stereo_pair is not None:
            pairs[stereo_pair.pair] = stereo_pair
            # logger.info(f"Successfully estimated pair {cam_id_a}-{cam_id_b} with RMSE: {stereo_pair.error_score:.6f}")
        else:
            logger.warning(f"Failed to estimate pair {cam_id_a}-{cam_id_b}")

    logger.info(f"Completed estimation for {len(pairs)} stereo pairs")
    return PairedPoseNetwork.from_raw_estimates(pairs)


def _add_coverage_regions(point_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute coverage regions for all points.
    Coverage region is a string like "_1_2_3_" showing which cam_ids see each point.
    """
    # Extract unique combinations of sync_index, point_id, and cam_id
    point_cam_ids = point_data[["sync_index", "point_id", "cam_id"]].drop_duplicates()

    # Convert cam_id to string for easier handling
    point_cam_ids["cam_id_str"] = point_cam_ids["cam_id"].astype(str)

    # Group by sync_index and point_id to collect cam_ids
    grouped = (
        point_cam_ids.groupby(["sync_index", "point_id"])["cam_id_str"]
        .apply(lambda x: "_" + "_".join(sorted(x)) + "_")
        .reset_index(name="coverage_region")
    )

    # Merge back with original data
    result = point_data.merge(grouped, on=["sync_index", "point_id"], how="left")

    return result


def _estimate_single_pair(
    points_with_coverage: pd.DataFrame,
    camera_array: CameraArray,
    cam_id_a: int,
    cam_id_b: int,
    boards_sampled: int,
) -> StereoPair | None:
    """
    Estimate extrinsics for a single camera pair using pre-computed coverage data.
    """
    # logger.info(f"Estimating stereo pair {cam_id_a}-{cam_id_b}...")

    # Get camera data
    cam_a = camera_array.cameras[cam_id_a]
    cam_b = camera_array.cameras[cam_id_b]

    if cam_a.matrix is None or cam_b.matrix is None:
        logger.warning(f"Camera {cam_id_a} or {cam_id_b} lacks intrinsics")
        return None

    # Filter points using pre-computed coverage regions
    a_str, b_str = str(cam_id_a), str(cam_id_b)
    in_region_a = points_with_coverage["coverage_region"].str.contains(f"_{a_str}_")
    in_region_b = points_with_coverage["coverage_region"].str.contains(f"_{b_str}_")
    in_cam_id_pair = points_with_coverage["cam_id"].isin([cam_id_a, cam_id_b])

    pair_points = points_with_coverage[in_region_a & in_region_b & in_cam_id_pair].copy()

    if pair_points.empty:
        logger.info(f"For pair {cam_id_a}-{cam_id_b} there are no shared points")
        return None

    # Count points per board and filter to boards with enough points
    board_counts = pair_points.groupby(["sync_index", "cam_id"]).size().reset_index(name="point_count")
    valid_boards = board_counts[board_counts["point_count"] >= 6]
    valid_boards_a = valid_boards[valid_boards["cam_id"] == cam_id_a]

    if valid_boards_a.empty:
        logger.info(f"For pair {cam_id_a}-{cam_id_b} there are no boards with sufficient points")
        return None

    # Sample boards deterministically
    sample_size = min(len(valid_boards_a), boards_sampled)

    if sample_size > 0:
        logger.info(f"Assembling {sample_size} shared boards for pair {cam_id_a}-{cam_id_b}")
        selected_boards = _select_diverse_boards(valid_boards_a, sample_size)
    else:
        logger.info(f"For pair {cam_id_a}-{cam_id_b} there are no shared boards")
        return None

    # Filter points to selected boards
    selected_sync = selected_boards["sync_index"].tolist()
    pair_points = pair_points[pair_points["sync_index"].isin(selected_sync)]

    # Prepare inputs for cv2.stereoCalibrate
    img_locs_a, obj_locs_a = _prepare_stereocal_inputs(cam_id_a, pair_points)
    img_locs_b, obj_locs_b = _prepare_stereocal_inputs(cam_id_b, pair_points)

    if not img_locs_a or not img_locs_b:
        logger.info(f"No calibration data prepared for pair {cam_id_a}-{cam_id_b}")
        return None

    # Undistort points
    norm_locs_a = [cam_a.undistort_points(pts, output="normalized") for pts in img_locs_a]
    norm_locs_b = [cam_b.undistort_points(pts, output="normalized") for pts in img_locs_b]

    # Stereo calibration
    stereocal_flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    K_perfect = np.identity(3)
    D_perfect = np.zeros(5)

    try:
        # Using normalized coordinates (K=identity, D=zeros) rather than pixel coordinates.
        # This allows a unified pipeline for fisheye and rectilinear lenses, and improves
        # numerical stability for bundle adjustment. In this reference frame, imageSize
        # is meaningless - there are no "pixels", just normalized ray directions.
        # OpenCV's type stubs don't account for this valid use case.
        ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            obj_locs_a,
            norm_locs_a,
            norm_locs_b,
            K_perfect,
            D_perfect,
            K_perfect,
            D_perfect,
            imageSize=None,  # type: ignore[arg-type]
            criteria=criteria,
            flags=stereocal_flags,
        )

        logger.info(f"Stereo calibration successful for pair {cam_id_a}-{cam_id_b}, RMSE: {ret:.6f}")

        return StereoPair(
            primary_cam_id=cam_id_a,
            secondary_cam_id=cam_id_b,
            error_score=float(ret),
            rotation=R,
            translation=T,
        )
    except Exception as e:
        logger.error(f"Stereo calibration failed for pair {cam_id_a}-{cam_id_b}: {e}")
        return None


def _select_diverse_boards(valid_boards_a: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Deterministically select boards with temporal and quality diversity.
    """
    # Sort by quality (point_count) descending, then sync_index for determinism
    boards_sorted = valid_boards_a.sort_values(["point_count", "sync_index"], ascending=[False, True]).reset_index(
        drop=True
    )

    if len(boards_sorted) <= sample_size:
        return boards_sorted

    # For temporal diversity, select boards spread across time range
    sync_indices = boards_sorted["sync_index"].to_numpy()
    min_sync, max_sync = sync_indices.min(), sync_indices.max()

    if max_sync > min_sync and sample_size > 1:
        time_bins = np.linspace(min_sync, max_sync + 1, sample_size + 1)
        selected_indices = []

        for i in range(sample_size):
            bin_start, bin_end = time_bins[i], time_bins[i + 1]
            bin_mask = (boards_sorted["sync_index"] >= bin_start) & (boards_sorted["sync_index"] < bin_end)
            bin_boards = boards_sorted[bin_mask]

            if not bin_boards.empty:
                selected_indices.append(bin_boards.index[0])

        # Fill remaining slots if needed
        remaining_needed = sample_size - len(selected_indices)
        if remaining_needed > 0:
            available = [idx for idx in boards_sorted.index if idx not in selected_indices]
            selected_indices.extend(available[:remaining_needed])

        return boards_sorted.loc[selected_indices[:sample_size]]

    # Fallback: just take top N by quality
    return boards_sorted.head(sample_size)


def _prepare_stereocal_inputs(cam_id: int, pair_points: pd.DataFrame):
    """
    Prepare image and object points for cv2.stereoCalibrate.
    """
    cam_data = pair_points[pair_points["cam_id"] == cam_id].copy()

    if cam_data.empty:
        return [], []

    # ensure deterministic output with explicit sort
    cam_data.sort_values(by=["sync_index", "point_id"], inplace=True)

    sync_indices = cam_data["sync_index"].to_numpy().round().astype(int)
    img_loc_x = cam_data["img_loc_x"].to_numpy().astype(np.float32)
    img_loc_y = cam_data["img_loc_y"].to_numpy().astype(np.float32)
    obj_loc_x = cam_data["obj_loc_x"].to_numpy().astype(np.float32)
    obj_loc_y = cam_data["obj_loc_y"].to_numpy().astype(np.float32)
    obj_loc_z = np.zeros_like(obj_loc_x)

    # Build arrays
    img_x_y = np.vstack([img_loc_x, img_loc_y]).T
    board_x_y_z = np.vstack([obj_loc_x, obj_loc_y, obj_loc_z]).T

    # Group by sync_index
    img_locs = []
    obj_locs = []
    for sync_index in np.unique(sync_indices):
        same_frame = sync_indices == sync_index
        img_locs.append(img_x_y[same_frame])
        obj_locs.append(board_x_y_z[same_frame])

    return img_locs, obj_locs
