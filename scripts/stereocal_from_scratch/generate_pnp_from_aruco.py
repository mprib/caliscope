"""
Validate PnP-based relative pose estimation against cv2.stereocalibrate ground truth.

Design Decisions:
- Uses SOLVEPNP_IPPE for planar Charuco boards (optimal for planar targets)
- Stores intermediate results in memory only (no pickling)
- Applies IQR-based outlier rejection to both rotation and translation
- Uses quaternion averaging for robust rotation aggregation
- Compares against gold standard using rotation angle and translation errors
- Calculates Stereo RMSE via triangulation/reprojection on the normalized plane
"""

import json
import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from caliscope import __root__
from caliscope.calibration.stereocalibrator import StereoCalibrator
from caliscope.cameras.camera_array import CameraArray
from caliscope.configurator import Configurator
from caliscope.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Minimum points for PnP (4 is minimum for non-planar, 6 is safer for noisy data)
MIN_PNP_POINTS = 4

# IQR multiplier for outlier rejection
OUTLIER_THRESHOLD = 1.5

# Number of boards to sample for gold standard (match stereocalibrator default)
GOLD_STANDARD_BOARDS = 10


def quaternion_average(quaternions: np.ndarray) -> np.ndarray:
    """
    Compute the average quaternion from a set of quaternions.

    Args:
        quaternions: (N, 4) array of quaternions (w, x, y, z)

    Returns:
        (4,) array representing the average quaternion (normalized)
    """
    if len(quaternions) == 0:
        raise ValueError("Cannot average empty quaternion array")

    if len(quaternions) == 1:
        return quaternions[0]

    # Compute the eigenvector corresponding to the largest eigenvalue
    # of the quaternion covariance matrix
    Q = quaternions.T
    M = Q @ Q.T
    _, eigenvecs = np.linalg.eigh(M)

    # The average quaternion is the eigenvector with largest eigenvalue
    avg_quat = eigenvecs[:, -1]

    # Normalize and ensure positive w component
    if avg_quat[0] < 0:
        avg_quat = -avg_quat

    return avg_quat / np.linalg.norm(avg_quat)


def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute rotation error in degrees between two rotation matrices.

    Args:
        R1, R2: (3, 3) rotation matrices

    Returns:
        Error in degrees
    """
    # Compute relative rotation
    R_rel = R1 @ R2.T

    # Convert to angle
    trace = np.trace(R_rel)
    # Clamp to valid range for arccos
    trace = np.clip(trace, -1.0, 3.0)
    angle = np.arccos((trace - 1) / 2)

    return np.degrees(angle)


def translation_error(t1: np.ndarray, t2: np.ndarray) -> dict:
    """
    Compute translation errors between two vectors.

    Args:
        t1, t2: (3,) translation vectors

    Returns:
        Dict with magnitude_error (%) and direction_error (degrees)
    """
    # Magnitude error as percentage
    mag1 = np.linalg.norm(t1)
    mag2 = np.linalg.norm(t2)

    if mag1 < 1e-10 or mag2 < 1e-10:
        magnitude_error = 0.0 if abs(mag1 - mag2) < 1e-10 else float("inf")
    else:
        magnitude_error = abs(mag1 - mag2) / mag1 * 100

    # Direction error in degrees
    if mag1 < 1e-10 or mag2 < 1e-10:
        direction_error = 0.0
    else:
        dot_product = np.dot(t1 / mag1, t2 / mag2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        direction_error = np.degrees(np.arccos(dot_product))

    return {"magnitude_error_pct": magnitude_error, "direction_error_deg": direction_error}


def load_point_data(point_data_path: Path) -> pd.DataFrame:
    """
    Load generic point data (e.g., Charuco or Keypoints) and add coverage regions.

    Returns:
        DataFrame with columns: sync_index, port, point_id, img_loc_x, img_loc_y,
        obj_loc_x, obj_loc_y, coverage_region
    """
    logger.info(f"Loading point data from {point_data_path}")

    raw_data = pd.read_csv(point_data_path)

    # Add coverage regions (same logic as stereocalibrator)
    point_ports = raw_data[["sync_index", "point_id", "port"]].drop_duplicates()
    point_ports["port_str"] = point_ports["port"].astype(str)

    grouped = (
        point_ports.groupby(["sync_index", "point_id"])["port_str"]
        .apply(lambda x: "_" + "_".join(sorted(x)) + "_")
        .rename("coverage_region")
        .reset_index()
    )

    result = raw_data.merge(grouped, on=["sync_index", "point_id"], how="left")
    logger.info(f"Loaded {len(result)} points across {result['sync_index'].nunique()} sync indices")

    return result


def compute_camera_poses_pnp(point_data: pd.DataFrame, camera_array: CameraArray) -> dict:
    """
    Compute per-camera poses using PnP for each sync_index.

    Returns:
        Dict mapping (port, sync_index) -> (R, t, reprojection_error)
    """
    logger.info("Computing per-frame camera poses with PnP...")

    poses = {}  # (port, sync_index) -> (R, t, rmse)
    success_count = 0
    failure_count = 0

    # Group by port and sync_index
    grouped = point_data.groupby(["port", "sync_index"])

    start_time = time.time()

    for key, group in grouped:
        # Type checker safe unpacking
        port, sync_index = key  # type: ignore

        # Check minimum point count
        if len(group) < MIN_PNP_POINTS:
            failure_count += 1
            continue

        # Extract 2D and 3D points
        img_points = group[["img_loc_x", "img_loc_y"]].to_numpy().astype(np.float32)
        obj_points = group[["obj_loc_x", "obj_loc_y"]].to_numpy()
        obj_points = np.hstack([obj_points, np.zeros((len(obj_points), 1))]).astype(np.float32)

        # Get camera intrinsics
        cam = camera_array.cameras[port]
        if cam.matrix is None:
            logger.warning(f"Camera {port} missing intrinsics, skipping")
            failure_count += 1
            continue

        # Undistort points (match stereocalibrator behavior)
        undistorted = cam.undistort_points(img_points)

        K_perfect = np.identity(3)
        D_perfect = np.zeros(5)

        # Run PnP
        # Use IPPE for planar targets, fallback to iterative if needed
        try:
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                undistorted,
                cameraMatrix=K_perfect,  # undistortion puts image points in "perfect" camera frame of reference.
                distCoeffs=D_perfect,  # distCoeffs already applied via undistort_points
                flags=cv2.SOLVEPNP_IPPE,
            )

            if not success:
                # Fallback to iterative method
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, undistorted, cameraMatrix=K_perfect, distCoeffs=D_perfect, flags=cv2.SOLVEPNP_ITERATIVE
                )

            if success:
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.flatten()

                # Compute reprojection error
                projected, _ = cv2.projectPoints(obj_points, rvec, tvec, K_perfect, D_perfect, None)
                projected = projected.reshape(-1, 2)
                rmse = np.sqrt(np.mean(np.sum((undistorted - projected) ** 2, axis=1)))

                poses[(port, sync_index)] = (R, t, rmse)
                success_count += 1
            else:
                failure_count += 1

        except Exception as e:
            logger.debug(f"PnP failed for port {port}, sync {sync_index}: {e}")
            failure_count += 1

    elapsed = time.time() - start_time
    logger.info(f"PnP complete: {success_count} successes, {failure_count} failures in {elapsed:.2f}s")
    logger.info(f"Average time per pose: {elapsed / max(success_count, 1) * 1000:.2f}ms")

    return poses


def compute_relative_poses(poses: dict, camera_array: CameraArray) -> dict:
    """
    Compute relative poses between camera pairs at each sync_index.

    Returns:
        Dict mapping (pair, sync_index) -> (R_rel, t_rel)
    """
    logger.info("Computing relative poses between camera pairs...")

    relative_poses = {}
    ports = [p for p in camera_array.cameras.keys() if not camera_array.cameras[p].ignore]
    pairs = [(i, j) for i, j in combinations(ports, 2) if i < j]

    # Find common sync indices for each pair
    for pair in pairs:
        port_a, port_b = pair

        # Get sync indices where both cameras have poses
        sync_a = {s for p, s in poses.keys() if p == port_a}
        sync_b = {s for p, s in poses.keys() if p == port_b}
        common_sync = sync_a.intersection(sync_b)

        for sync_index in common_sync:
            R_a, t_a, _ = poses[(port_a, sync_index)]
            R_b, t_b, _ = poses[(port_b, sync_index)]

            # We want to find the transform from camera A to camera B (T_ba)
            # T_ba = T_b_obj * inv(T_a_obj)
            # R_ba = R_b * R_a.T
            # t_ba = t_b - R_b * R_a.T * t_a

            R_a_inv = R_a.T
            t_a_inv = -R_a_inv @ t_a

            # Now compose the transformations: T_b_obj * T_obj_a
            R_rel = R_b @ R_a_inv
            t_rel = R_b @ t_a_inv + t_b

            relative_poses[(pair, sync_index)] = (R_rel, t_rel)

    logger.info(f"Computed {len(relative_poses)} relative poses across {len(pairs)} pairs")
    return relative_poses


def reject_outliers(relative_poses: dict) -> dict:
    """
    Apply IQR-based outlier rejection to relative poses.

    Returns:
        Dict mapping pair -> list of (R, t) that are not outliers
    """
    logger.info("Applying outlier rejection...")

    # Group by pair
    poses_by_pair = {}
    for (pair, sync_index), (R, t) in relative_poses.items():
        if pair not in poses_by_pair:
            poses_by_pair[pair] = []
        poses_by_pair[pair].append((R, t, sync_index))

    filtered_poses = {}

    for pair, poses_list in poses_by_pair.items():
        valid_poses = [
            (R, t, sync_index) for R, t, sync_index in poses_list if not (np.any(np.isnan(R)) or np.any(np.isnan(t)))
        ]

        poses_list = valid_poses

        if len(poses_list) < 5:
            # Too few samples for reliable outlier detection
            logger.warning(f"Pair {pair} has only {len(poses_list)} samples, skipping outlier rejection")
            filtered_poses[pair] = [(R, t) for R, t, _ in poses_list]
            continue

        # Convert to arrays for analysis
        quats = []
        t_mags = []

        for R, t, _ in poses_list:
            # Convert rotation to quaternion
            quat = Rotation.from_matrix(R).as_quat()  # Returns (x, y, z, w)
            quat = np.roll(quat, 1)  # Convert to (w, x, y, z) for our averaging function
            quats.append(quat)
            t_mags.append(np.linalg.norm(t))

        quats = np.array(quats)
        t_mags = np.array(t_mags)

        # IQR-based outlier detection for translation magnitude
        t_q1, t_q3 = np.percentile(t_mags, [25, 75])
        t_iqr = t_q3 - t_q1
        t_lower = t_q1 - OUTLIER_THRESHOLD * t_iqr
        t_upper = t_q3 + OUTLIER_THRESHOLD * t_iqr

        # For rotation, use quaternion distance from median
        median_quat = quaternion_average(quats)
        # Compute angular distance from median
        R_median = Rotation.from_quat(np.roll(median_quat, -1)).as_matrix()
        rot_angles = [rotation_error(R, R_median) for R, _, _ in poses_list]
        rot_angles = np.array(rot_angles)

        rot_q1, rot_q3 = np.percentile(rot_angles, [25, 75])
        rot_iqr = rot_q3 - rot_q1
        rot_upper = rot_q3 + OUTLIER_THRESHOLD * rot_iqr

        # Filter outliers
        filtered = []
        outlier_count = 0

        for i, (R, t, sync_index) in enumerate(poses_list):
            is_t_outlier = t_mags[i] < t_lower or t_mags[i] > t_upper
            is_rot_outlier = rot_angles[i] > rot_upper

            if not (is_t_outlier or is_rot_outlier):
                filtered.append((R, t))
            else:
                outlier_count += 1
                logger.debug(f"Outlier detected for pair {pair}, sync {sync_index}")

        logger.info(f"Pair {pair}: {outlier_count}/{len(poses_list)} outliers rejected")
        filtered_poses[pair] = filtered

    return filtered_poses


def aggregate_poses(filtered_poses: dict) -> dict:
    """
    Average poses after outlier rejection.

    Returns:
        Dict mapping pair -> (R_avg, t_avg)
    """
    logger.info("Aggregating poses...")

    aggregated = {}

    for pair, poses_list in filtered_poses.items():
        if not poses_list:
            logger.warning(f"No valid poses for pair {pair} after outlier rejection")
            continue

        if len(poses_list) == 1:
            aggregated[pair] = poses_list[0]
            continue

        # Convert rotations to quaternions
        quats = []
        translations = []

        for R, t in poses_list:
            quat = Rotation.from_matrix(R).as_quat()
            quat = np.roll(quat, 1)  # Convert to (w, x, y, z)
            quats.append(quat)
            translations.append(t)

        # Average quaternions
        avg_quat = quaternion_average(np.array(quats))

        # Average translations
        avg_translation = np.mean(translations, axis=0)

        # Convert back to rotation matrix
        avg_R = Rotation.from_quat(np.roll(avg_quat, -1)).as_matrix()

        aggregated[pair] = (avg_R, avg_translation)

    logger.info(f"Aggregated poses for {len(aggregated)} pairs")
    return aggregated


def calculate_stereo_rmse(
    pair: Tuple[int, int], R: np.ndarray, t: np.ndarray, camera_array: CameraArray, point_data: pd.DataFrame
) -> float | None:
    """
    Calculate the RMSE for a stereo pair given a fixed relative pose (R, t).
    mimics cv2.stereoCalibrate internal error calculation:
    1. Undistort points to normalized plane
    2. Triangulate using the provided pose
    3. Project back to both cameras
    4. Calculate RMS error of residuals

    Args:
        pair: tuple of (port_A, port_B)
        R: Rotation matrix from A to B
        t: Translation vector from A to B
        camera_array: CameraArray object with intrinsics
        point_data: DataFrame with columns [port, sync_index, point_id, img_loc_x, img_loc_y]

    Returns:
        RMSE value (float) or None if insufficient data
    """
    port_A, port_B = pair
    cam_A = camera_array.cameras[port_A]
    cam_B = camera_array.cameras[port_B]

    # Filter data for this pair
    data_A = point_data[point_data["port"] == port_A]
    data_B = point_data[point_data["port"] == port_B]

    # Merge to find common points
    common = pd.merge(
        data_A,
        data_B,
        on=["sync_index", "point_id"],
        suffixes=("_A", "_B"),
    )

    if len(common) < MIN_PNP_POINTS:
        logger.warning(f"Insufficient common points for RMSE calc on pair {pair}")
        return None

    # Extract points
    pts_A = common[["img_loc_x_A", "img_loc_y_A"]].to_numpy(dtype=np.float32)
    pts_B = common[["img_loc_x_B", "img_loc_y_B"]].to_numpy(dtype=np.float32)

    # 1. Undistort points to normalized plane (Camera Matrix = I, Dist = 0)
    norm_A = cam_A.undistort_points(pts_A)
    norm_B = cam_B.undistort_points(pts_B)

    # 2. Triangulate
    # Projection matrix for A (Origin): [I | 0]
    P1 = np.eye(3, 4)
    # Projection matrix for B (Relative): [R | t]
    P2 = np.hstack((R, t.reshape(3, 1)))

    # OpenCV triangulatePoints requires 2xN arrays
    points_4d = cv2.triangulatePoints(P1, P2, norm_A.T, norm_B.T)
    points_3d = points_4d[:3] / points_4d[3]  # Convert homogeneous to Euclidean (3xN)

    # 3. Project back
    # We project the triangulated 3D points back to the NORMALIZED image plane
    # So we use K=Identity, D=Zero.

    # Project to Camera A (Identity pose)
    # note: need to better explain and document "mystery params" here.
    proj_A, _ = cv2.projectPoints(points_3d.T, np.zeros(3), np.zeros(3), np.eye(3), np.zeros(5))
    proj_A = proj_A.reshape(-1, 2)

    # Project to Camera B (Relative pose)
    # Note: projectPoints expects rvec, not rotation matrix
    rvec, _ = cv2.Rodrigues(R)
    proj_B, _ = cv2.projectPoints(points_3d.T, rvec, t, np.eye(3), np.zeros(5))
    proj_B = proj_B.reshape(-1, 2)

    # 4. Calculate RMSE
    err_A = norm_A - proj_A
    err_B = norm_B - proj_B

    # Stack errors from both cameras
    all_errors = np.vstack((err_A, err_B))

    # RMSE = sqrt( sum(d^2) / N )
    residuals_squared = np.sum(all_errors**2, axis=1)  # squared euclidean distance per point
    mean_sq_error = np.mean(residuals_squared)
    rmse = np.sqrt(mean_sq_error)

    return float(rmse)


def compare_to_gold_standard(aggregated_poses: dict, gold_standard: dict, output_dir: Path) -> pd.DataFrame:
    """
    Compare aggregated PnP poses to gold standard and generate comparison table.

    Returns:
        DataFrame with comparison metrics
    """
    logger.info("Comparing to gold standard...")

    comparison_rows = []

    for pair, (R_pnp, t_pnp) in aggregated_poses.items():
        pair_key = f"stereo_{pair[0]}_{pair[1]}"

        if pair_key not in gold_standard:
            logger.warning(f"Gold standard not found for {pair_key}")
            continue

        gs_data = gold_standard[pair_key]
        R_gs = np.array(gs_data["rotation"])
        t_gs = np.array(gs_data["translation"]).flatten()

        # Compute errors
        rot_err = rotation_error(R_pnp, R_gs)
        trans_err = translation_error(t_pnp, t_gs)

        comparison_rows.append(
            {
                "pair": pair_key,
                "port_a": pair[0],
                "port_b": pair[1],
                "rotation_error_deg": rot_err,
                "translation_magnitude_error_pct": trans_err["magnitude_error_pct"],
                "translation_direction_error_deg": trans_err["direction_error_deg"],
                "pnp_translation_norm": np.linalg.norm(t_pnp),
                "gold_translation_norm": np.linalg.norm(t_gs),
                "relative_translation_diff": np.linalg.norm(t_pnp - t_gs),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    # Save to CSV
    output_file = output_dir / "comparison_table.csv"
    comparison_df.to_csv(output_file, index=False)
    logger.info(f"Comparison table saved to {output_file}")

    # Log summary statistics
    logger.info("=" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Mean rotation error: {comparison_df['rotation_error_deg'].mean():.4f}°")
    logger.info(f"Std rotation error: {comparison_df['rotation_error_deg'].std():.4f}°")
    logger.info(f"Mean translation magnitude error: {comparison_df['translation_magnitude_error_pct'].mean():.2f}%")
    logger.info(f"Mean translation direction error: {comparison_df['translation_direction_error_deg'].mean():.4f}°")
    logger.info(f"Max rotation error: {comparison_df['rotation_error_deg'].max():.4f}°")
    logger.info(f"Max translation magnitude error: {comparison_df['translation_magnitude_error_pct'].max():.2f}%")

    return comparison_df


def main():
    """Main validation pipeline."""
    logger.info("Starting PnP validation pipeline...")

    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # test_data_dir = __root__ / "tests/sessions/post_optimization"
    project_fixture_dir = __root__ / "scripts/fixtures/aruco_pipeline"
    calibration_video_dir = project_fixture_dir / "calibration/extrinsic"
    # NOTE: keeping file name for compatibility, but treating as generic point data
    point_data_file = calibration_video_dir / "CHARUCO/xy_CHARUCO.csv"

    # NOTE: project fixture dir created with the following
    # copy_contents(test_data_dir,project_fixture_dir)

    # Load config and camera array
    config = Configurator(project_fixture_dir)
    camera_array: CameraArray = config.get_camera_array()

    stages_to_run = [1, 2, 3]
    gold_standard = None  # Created in Step 1
    relative_poses = None  # Created in Step 2
    point_data = None

    # Stage 1: get stereo poses based on cv2.stereocalibrate and ChAruco
    if 1 in stages_to_run:
        # Stage 1: Generate gold standard
        logger.info("=" * 20 + " STAGE 1: Gold Standard Generation " + "=" * 20)
        stereocal = StereoCalibrator(camera_array=camera_array, point_data_path=point_data_file)
        gold_standard = stereocal.stereo_calibrate_all(boards_sampled=GOLD_STANDARD_BOARDS)

        # Save gold standard
        gold_file = output_dir / "gold_standard.json"
        with open(gold_file, "w") as f:
            json.dump(gold_standard, f, indent=2)
        logger.info(f"Gold standard saved to {gold_file}")

    # Stage 2: Compute relative poses
    if 2 in stages_to_run:
        # Load Point data
        logger.info("=" * 20 + " STAGE 2A: Data Loading " + "=" * 20)
        point_data = load_point_data(point_data_file)

        # Compute per-camera poses with PnP
        logger.info("=" * 20 + " STAGE 2B: Per-Frame PnP " + "=" * 20)
        camera_poses = compute_camera_poses_pnp(point_data, camera_array)

        # Compute relative poses
        logger.info("=" * 20 + " STAGE 2C: Relative Poses " + "=" * 20)
        relative_poses = compute_relative_poses(camera_poses, camera_array)

    # Stage 3: compare gold standard and PnP results
    if 3 in stages_to_run:
        if gold_standard is None:
            raise RuntimeError("need to run stage 1")

        if relative_poses is None:
            raise RuntimeError("need to run stage 2")

        if point_data is None:
            # fallback load if stage 2 didn't run but we want to run stage 3
            # (in this script logic they are sequential, but good for safety)
            point_data = load_point_data(point_data_file)

        # Stage 3: Outlier rejection
        logger.info("=" * 20 + " STAGE 3A: Outlier Rejection " + "=" * 20)
        filtered_poses = reject_outliers(relative_poses)

        logger.info("=" * 20 + " STAGE 3B: Pose Aggregation " + "=" * 20)
        aggregated_poses = aggregate_poses(filtered_poses)

        # Stage 3C: RMSE Calculation and Export
        logger.info("=" * 20 + " STAGE 3C: RMSE Calculation & Export " + "=" * 20)
        pnp_estimates = {}
        for pair, (R, t) in aggregated_poses.items():
            rmse = calculate_stereo_rmse(pair, R, t, camera_array, point_data)

            pair_key = f"stereo_{pair[0]}_{pair[1]}"
            pnp_estimates[pair_key] = {
                "rotation": R.tolist(),
                "translation": t.tolist(),
                "RMSE": rmse,
            }
            logger.info(f"Pair {pair}: RMSE = {rmse:.5f} (PnP)")

        pnp_file = output_dir / "pnp_estimates.json"
        with open(pnp_file, "w") as f:
            json.dump(pnp_estimates, f, indent=2)
        logger.info(f"PnP estimates with RMSE saved to {pnp_file}")

        # Stage 7: Comparison
        logger.info("=" * 20 + " STAGE 3D: Gold Standard Comparison " + "=" * 20)
        compare_to_gold_standard(aggregated_poses, gold_standard, output_dir)

        logger.info("=" * 20 + " VALIDATION COMPLETE " + "=" * 20)
        logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    print("Start")
    main()
    print("Stop")
