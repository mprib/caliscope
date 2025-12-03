"""
PnP-based Relative Pose Estimation for Camera Array Initialization.

This script provides a clean, modular workflow for estimating camera pair extrinsics
using 3D object keypoints and PnP, as an alternative to cv2.stereocalibrate. The
resulting poses are packaged into a PairedPoseNetwork for compatibility with the
existing calibration pipeline.

Design Decisions:
- Uses SOLVEPNP_IPPE for planar targets (optimal) with iterative fallback
- Undistorts points once at the PnP stage to work in normalized coordinates
- Applies IQR-based outlier rejection to translation magnitude and rotation angle
- Aggregates poses via quaternion averaging (robust to rotation space non-linearity)
- Validates against gold standard using rotation angle and translation errors
- Computes Stereo RMSE via triangulation/reprojection on the normalized plane
"""

import json
import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from caliscope import __root__
from caliscope.calibration.array_initialization.paired_pose_network import PairedPoseNetwork
from caliscope.calibration.array_initialization.stereopairs import StereoPair
from caliscope.calibration.array_initialization.estimate_paired_pose_network import (
    estimate_paired_pose_network,
)
from caliscope.cameras.camera_array import CameraArray
from caliscope.configurator import Configurator
from caliscope.logger import setup_logging
from caliscope.post_processing.point_data import ImagePoints

setup_logging()
logger = logging.getLogger(__name__)

# Minimum points for reliable PnP (4 is theoretical minimum, 6 is safer for noisy data)
MIN_PNP_POINTS = 4

# IQR multiplier for outlier rejection (1.5 is standard for box plots)
OUTLIER_THRESHOLD = 1.5

# Number of boards to sample for gold standard (matches stereocalibrator default)
GOLD_STANDARD_BOARDS = 10


def quaternion_average(quaternions: np.ndarray) -> np.ndarray:
    """
    Compute the robust average quaternion from a set of quaternions.

    Uses the eigenvector method which is optimal for small dispersion. Handles
    antipodal ambiguity by ensuring positive w component.

    Args:
        quaternions: (N, 4) array of quaternions in (w, x, y, z) format

    Returns:
        (4,) array representing the average quaternion (normalized)
    """
    if len(quaternions) == 0:
        raise ValueError("Cannot average empty quaternion array")
    if len(quaternions) == 1:
        return quaternions[0]

    # Compute eigenvector of covariance matrix for largest eigenvalue
    Q = quaternions.T
    M = Q @ Q.T
    _, eigenvecs = np.linalg.eigh(M)

    avg_quat = eigenvecs[:, -1]
    # Ensure positive w for consistency
    if avg_quat[0] < 0:
        avg_quat = -avg_quat

    return avg_quat / np.linalg.norm(avg_quat)


def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute geodesic rotation error in degrees between two rotation matrices.

    This is the minimal rotation angle needed to align R1 with R2.

    Args:
        R1, R2: (3, 3) rotation matrices

    Returns:
        Error in degrees
    """
    R_rel = R1 @ R2.T
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)  # Clamp for numerical stability
    angle = np.arccos((trace - 1) / 2)
    return np.degrees(angle)


def translation_error(t1: np.ndarray, t2: np.ndarray) -> dict[str, float]:
    """
    Compute translation errors between two vectors.

    Args:
        t1, t2: (3,) translation vectors

    Returns:
        Dict with magnitude_error (%) and direction_error (degrees)
    """
    mag1, mag2 = np.linalg.norm(t1), np.linalg.norm(t2)

    # Handle near-zero magnitudes
    if mag1 < 1e-10 or mag2 < 1e-10:
        magnitude_error = 0.0 if abs(mag1 - mag2) < 1e-10 else float("inf")
        direction_error = 0.0
    else:
        magnitude_error = abs(mag1 - mag2) / mag1 * 100
        dot_product = np.clip(np.dot(t1 / mag1, t2 / mag2), -1.0, 1.0)
        direction_error = np.degrees(np.arccos(dot_product))

    return {"magnitude_delta_pct": magnitude_error, "direction_delta_deg": direction_error}


def load_point_data(point_data_path: Path) -> pd.DataFrame:
    """
    Load generic point data and pre-compute coverage regions.

    Coverage regions (e.g., "_1_2_3_") indicate which cameras see each point
    at each sync index, enabling efficient pair filtering.

    Returns:
        DataFrame with coverage_region column added
    """
    logger.info(f"Loading point data from {point_data_path}")
    raw_data = pd.read_csv(point_data_path)

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


def compute_camera_to_object_poses_pnp(
    image_points: ImagePoints, camera_array: CameraArray
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float]]:
    """
    Compute per-camera poses relative to the object coordinate frame for each sync_index.

    Uses PnP with undistorted points in the normalized image plane. SOLVEPNP_IPPE
    is preferred for planar targets but falls back to iterative if needed.

    Returns:
        Dict mapping (port, sync_index) -> (R, t, reprojection_error)
    """
    logger.info("Computing per-frame camera poses with PnP...")
    poses = {}
    success_count = 0
    failure_count = 0

    grouped = image_points.df.groupby(["port", "sync_index"])

    start_time = time.time()
    for (port, sync_index), group in grouped:
        if len(group) < MIN_PNP_POINTS:
            failure_count += 1
            continue

        # Extract 2D and 3D correspondences
        img_points = group[["img_loc_x", "img_loc_y"]].to_numpy(dtype=np.float32)
        obj_points = group[["obj_loc_x", "obj_loc_y"]].to_numpy()
        obj_points = np.hstack([obj_points, np.zeros((len(obj_points), 1))]).astype(np.float32)

        camera = camera_array.cameras[port]
        if camera.matrix is None:
            logger.warning(f"Camera {port} missing intrinsics, skipping")
            failure_count += 1
            continue

        # Work in normalized coordinates after undistortion
        undistorted = camera.undistort_points(img_points)
        K_perfect = np.identity(3)
        D_perfect = np.zeros(5)

        # PnP with IPPE for planar targets, fallback to iterative
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            undistorted,
            cameraMatrix=K_perfect,
            distCoeffs=D_perfect,
            flags=cv2.SOLVEPNP_IPPE,
        )

        if not success:
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                undistorted,
                cameraMatrix=K_perfect,
                distCoeffs=D_perfect,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if success:
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()

            # Compute reprojection error in normalized space
            projected, _ = cv2.projectPoints(obj_points, rvec, tvec, K_perfect, D_perfect)
            rmse = np.sqrt(np.mean(np.sum((undistorted - projected.reshape(-1, 2)) ** 2, axis=1)))

            poses[(port, sync_index)] = (R, t, rmse)
            success_count += 1
        else:
            failure_count += 1

    elapsed = time.time() - start_time
    logger.info(
        f"PnP complete: {success_count} successes, {failure_count} failures "
        f"in {elapsed:.2f}s ({elapsed / max(success_count, 1) * 1000:.2f}ms avg)"
    )
    return poses


def compute_relative_poses(
    camera_to_object_poses: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float]],
    camera_array: CameraArray,
) -> Dict[Tuple[Tuple[int, int], int], StereoPair]:
    """
    Compute relative poses between camera pairs at each sync_index.

    For cameras A and B observing the same object at a sync index:
    T_B_A = T_B_obj @ T_obj_A = T_B_obj @ inv(T_A_obj)

    Returns:
        Dict mapping (pair, sync_index) -> StereoPair with relative pose
    """
    logger.info("Computing relative poses between camera pairs...")
    relative_poses = {}

    ports = [p for p, cam in camera_array.cameras.items() if not cam.ignore]
    pairs = [(i, j) for i, j in combinations(ports, 2) if i < j]

    for port_a, port_b in pairs:
        # Find sync indices where both cameras have poses
        sync_a = {s for p, s in camera_to_object_poses.keys() if p == port_a}
        sync_b = {s for p, s in camera_to_object_poses.keys() if p == port_b}
        common_sync = sync_a.intersection(sync_b)

        for sync_index in common_sync:
            R_a, t_a, _ = camera_to_object_poses[(port_a, sync_index)]
            R_b, t_b, _ = camera_to_object_poses[(port_b, sync_index)]

            # Compute relative transformation: T_B_A = T_B_obj @ inv(T_A_obj)
            R_a_inv = R_a.T
            t_a_inv = -R_a_inv @ t_a
            R_rel = R_b @ R_a_inv
            t_rel = R_b @ t_a_inv + t_b

            # Store as StereoPair (error_score will be computed after aggregation)
            pair_key = (port_a, port_b)
            relative_poses[(pair_key, sync_index)] = StereoPair(
                primary_port=port_a,
                secondary_port=port_b,
                error_score=None,  # Will be filled after RMSE calculation
                translation=t_rel,
                rotation=R_rel,
            )

    logger.info(f"Computed {len(relative_poses)} relative poses across {len(pairs)} pairs")
    return relative_poses


def reject_outliers(
    relative_poses: Dict[Tuple[Tuple[int, int], int], StereoPair],
) -> Dict[Tuple[int, int], list[StereoPair]]:
    """
    Apply IQR-based outlier rejection to relative poses for each camera pair.

    Rejects poses based on translation magnitude and rotation angle from the
    median pose. Operates independently per pair to handle varying baselines.

    Returns:
        Dict mapping pair -> list of StereoPair that passed outlier rejection
    """
    logger.info("Applying outlier rejection...")

    # Group poses by pair
    poses_by_pair: Dict[Tuple[int, int], list[StereoPair]] = {}
    for (pair, _sync_index), stereo_pair in relative_poses.items():
        poses_by_pair.setdefault(pair, []).append(stereo_pair)

    filtered_poses = {}
    for pair, stereo_pairs in poses_by_pair.items():
        # Filter NaN poses (shouldn't happen but safer to check)
        valid_pairs = [
            sp for sp in stereo_pairs if not (np.any(np.isnan(sp.rotation)) or np.any(np.isnan(sp.translation)))
        ]

        if len(valid_pairs) < 5:
            logger.warning(f"Pair {pair} has only {len(valid_pairs)} samples, skipping outlier rejection")
            filtered_poses[pair] = valid_pairs
            continue

        # Extract arrays for statistical analysis
        quats = []
        t_mags = []
        for sp in valid_pairs:
            quat = Rotation.from_matrix(sp.rotation).as_quat()  # (x, y, z, w)
            quats.append(np.roll(quat, 1))  # Convert to (w, x, y, z)
            t_mags.append(np.linalg.norm(sp.translation))

        quats = np.array(quats)
        t_mags = np.array(t_mags)

        # Translation magnitude IQR filter
        t_q1, t_q3 = np.percentile(t_mags, [25, 75])
        t_iqr = t_q3 - t_q1
        t_lower, t_upper = t_q1 - OUTLIER_THRESHOLD * t_iqr, t_q3 + OUTLIER_THRESHOLD * t_iqr

        # Rotation angle IQR filter (angle from median quaternion)
        median_quat = quaternion_average(quats)
        R_median = Rotation.from_quat(np.roll(median_quat, -1)).as_matrix()
        rot_angles = np.array([rotation_error(sp.rotation, R_median) for sp in valid_pairs])

        rot_q1, rot_q3 = np.percentile(rot_angles, [25, 75])
        rot_iqr = rot_q3 - rot_q1
        rot_upper = rot_q3 + OUTLIER_THRESHOLD * rot_iqr

        # Apply filters
        filtered = []
        outlier_count = 0
        for i, stereo_pair in enumerate(valid_pairs):
            is_t_outlier = t_mags[i] < t_lower or t_mags[i] > t_upper
            is_rot_outlier = rot_angles[i] > rot_upper

            if not (is_t_outlier or is_rot_outlier):
                filtered.append(stereo_pair)
            else:
                outlier_count += 1

        logger.info(f"Pair {pair}: {outlier_count}/{len(valid_pairs)} outliers rejected")
        filtered_poses[pair] = filtered

    return filtered_poses


def aggregate_poses(filtered_poses: Dict[Tuple[int, int], list[StereoPair]]) -> Dict[Tuple[int, int], StereoPair]:
    """
    Aggregate per-sync-index poses into a single robust estimate per pair.

    Uses quaternion averaging for rotations and simple averaging for translations
    after outlier rejection.

    Returns:
        Dict mapping pair -> aggregated StereoPair
    """
    logger.info("Aggregating poses...")

    aggregated = {}
    for pair, stereo_pairs in filtered_poses.items():
        if not stereo_pairs:
            logger.warning(f"No valid poses for pair {pair} after outlier rejection")
            continue

        if len(stereo_pairs) == 1:
            aggregated[pair] = stereo_pairs[0]
            continue

        # Extract and average rotations (via quaternion space)
        quats = [np.roll(Rotation.from_matrix(sp.rotation).as_quat(), 1) for sp in stereo_pairs]
        avg_quat = quaternion_average(np.array(quats))
        avg_R = Rotation.from_quat(np.roll(avg_quat, -1)).as_matrix()

        # Average translations
        translations = [sp.translation for sp in stereo_pairs]
        avg_t = np.mean(translations, axis=0)

        aggregated[pair] = StereoPair(
            primary_port=pair[0],
            secondary_port=pair[1],
            error_score=None,  # Will be filled by RMSE calculation
            rotation=avg_R,
            translation=avg_t,
        )

    logger.info(f"Aggregated poses for {len(aggregated)} pairs")
    return aggregated


def calculate_stereo_rmse_for_pair(
    stereo_pair: StereoPair, camera_array: CameraArray, point_data: pd.DataFrame
) -> float | None:
    """
    Calculate Stereo RMSE for a pair using triangulation/reprojection error.

    Mimics cv2.stereoCalibrate's internal error calculation:
    1. Undistort points to normalized plane
    2. Triangulate using the provided pose
    3. Project back to both cameras
    4. Calculate RMS error of residuals

    Returns:
        RMSE value or None if insufficient data
    """
    port_a, port_b = stereo_pair.pair
    cam_a, cam_b = camera_array.cameras[port_a], camera_array.cameras[port_b]

    # Find common observations
    data_a = point_data[point_data["port"] == port_a]
    data_b = point_data[point_data["port"] == port_b]

    common = pd.merge(data_a, data_b, on=["sync_index", "point_id"], suffixes=("_a", "_b"))
    if len(common) < MIN_PNP_POINTS:
        logger.warning(f"Insufficient common points for RMSE calc on pair {stereo_pair}")
        return None

    # Extract and undistort points
    pts_a = common[["img_loc_x_a", "img_loc_y_a"]].to_numpy(dtype=np.float32)
    pts_b = common[["img_loc_x_b", "img_loc_y_b"]].to_numpy(dtype=np.float32)
    norm_a = cam_a.undistort_points(pts_a)
    norm_b = cam_b.undistort_points(pts_b)

    # Triangulate using the stereo pair pose
    P1 = np.eye(3, 4)  # Camera A at origin
    P2 = np.hstack((stereo_pair.rotation, stereo_pair.translation.reshape(3, 1)))  # Camera B relative to A

    points_4d = cv2.triangulatePoints(P1, P2, norm_a.T, norm_b.T)
    points_3d = points_4d[:3] / points_4d[3]

    # Project back to normalized plane and compute residuals
    proj_a, _ = cv2.projectPoints(points_3d.T, np.zeros(3), np.zeros(3), np.eye(3), np.zeros(5))
    proj_b, _ = cv2.projectPoints(
        points_3d.T, cv2.Rodrigues(stereo_pair.rotation)[0], stereo_pair.translation, np.eye(3), np.zeros(5)
    )

    errors = np.vstack([norm_a - proj_a.reshape(-1, 2), norm_b - proj_b.reshape(-1, 2)])
    rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

    return float(rmse)


def estimate_pnp_paired_pose_network(
    aggregated_poses: Dict[Tuple[int, int], StereoPair], camera_array: CameraArray, point_data: pd.DataFrame
) -> PairedPoseNetwork:
    """
    Create a PairedPoseNetwork from aggregated stereo pairs with RMSE scores.

    Calculates the Stereo RMSE for each pair and populates the error_score field.

    Returns:
        PairedPoseNetwork ready for use in camera array initialization
    """
    logger.info("Creating PairedPoseNetwork...")

    pairs_with_rmse = {}
    for pair, stereo_pair in aggregated_poses.items():
        rmse = calculate_stereo_rmse_for_pair(stereo_pair, camera_array, point_data)
        if rmse is None:
            logger.warning(f"Could not compute RMSE for pair {pair}, skipping")
            continue

        # Create new StereoPair with RMSE populated
        pairs_with_rmse[pair] = StereoPair(
            primary_port=stereo_pair.primary_port,
            secondary_port=stereo_pair.secondary_port,
            error_score=rmse,
            rotation=stereo_pair.rotation,
            translation=stereo_pair.translation,
        )
        logger.info(f"Pair {pair}: RMSE = {rmse:.6f}")

    return PairedPoseNetwork.from_raw_estimates(pairs_with_rmse)


def compare_to_gold_standard(
    pnp_network: PairedPoseNetwork, gold_standard_network: PairedPoseNetwork, output_dir: Path
) -> pd.DataFrame:
    """
    Compare PnP-based PairedPoseNetwork to gold standard and generate metrics.

    Returns:
        DataFrame with comparison metrics for each pair
    """
    logger.info("Comparing to gold standard...")

    comparison_rows = []
    for pair, pnp_pair in pnp_network._pairs.items():
        if pair not in gold_standard_network._pairs:
            logger.warning(f"Gold standard not found for pair {pair}")
            continue

        gs_pair = gold_standard_network._pairs[pair]

        rot_err = rotation_error(pnp_pair.rotation, gs_pair.rotation)
        trans_err = translation_error(pnp_pair.translation, gs_pair.translation)

        comparison_rows.append(
            {
                "pair": f"stereo_{pair[0]}_{pair[1]}",
                "port_a": pair[0],
                "port_b": pair[1],
                "rotation_delta_deg": rot_err,
                "translation_magnitude_delta_pct": trans_err["magnitude_delta_pct"],
                "translation_direction_delta_deg": trans_err["direction_delta_deg"],
                "pnp_translation_norm": np.linalg.norm(pnp_pair.translation),
                "gold_translation_norm": np.linalg.norm(gs_pair.translation),
                "relative_translation_diff": np.linalg.norm(pnp_pair.translation - gs_pair.translation),
                "pnp_rmse": pnp_pair.error_score,
                "gold_rmse": gs_pair.error_score,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    output_file = output_dir / "comparison_table.csv"
    comparison_df.to_csv(output_file, index=False)
    logger.info(f"Comparison table saved to {output_file}")

    # Log summary statistics
    logger.info("=" * 50)
    logger.info("VALIDATION SUMMARY: COMPARISON WITH GOLD STANDARD")
    logger.info("=" * 50)
    logger.info(f"Mean rotation delta {comparison_df['rotation_delta_deg'].mean():.4f}°")
    logger.info(f"Std rotation delta {comparison_df['rotation_delta_deg'].std():.4f}°")
    logger.info(f"Mean translation magnitude delta {comparison_df['translation_magnitude_delta_pct'].mean():.2f}%")
    logger.info(f"Mean translation direction delta {comparison_df['translation_direction_delta_deg'].mean():.4f}°")
    logger.info(f"Max rotation delta {comparison_df['rotation_delta_deg'].max():.4f}°")
    logger.info(f"Max translation magnitude delta {comparison_df['translation_magnitude_delta_pct'].max():.2f}%")

    return comparison_df


def save_network_to_json(network: PairedPoseNetwork, output_path: Path) -> None:
    """Serialize a PairedPoseNetwork to JSON for inspection/debugging."""
    data = {}
    for pair, stereo_pair in network._pairs.items():
        data[f"stereo_{pair[0]}_{pair[1]}"] = {
            "primary_port": stereo_pair.primary_port,
            "secondary_port": stereo_pair.secondary_port,
            "error_score": stereo_pair.error_score,
            "rotation": stereo_pair.rotation.tolist(),
            "translation": stereo_pair.translation.tolist(),
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Network saved to {output_path}")


def main():
    """Main validation pipeline comparing PnP to stereocalibrate gold standard."""
    logger.info("Starting PnP validation pipeline...")

    script_dir = Path(__file__).parent
    output_dir = script_dir / "working_output"
    output_dir.mkdir(exist_ok=True)

    project_fixture_dir = __root__ / "scripts/stereocal_from_scratch/aruco_pipeline"
    calibration_video_dir = project_fixture_dir / "calibration/extrinsic"
    charuco_point_data_file = calibration_video_dir / "CHARUCO/xy_CHARUCO.csv"

    config = Configurator(project_fixture_dir)
    camera_array = config.get_camera_array()
    image_points = ImagePoints.from_csv(charuco_point_data_file)

    # Stage 1: Generate gold standard using legacy stereocalibrate
    logger.info("=" * 20 + " STAGE 1: Gold Standard Generation " + "=" * 20)
    gold_standard_network = estimate_paired_pose_network(
        image_points, camera_array, boards_sampled=GOLD_STANDARD_BOARDS
    )
    save_network_to_json(gold_standard_network, output_dir / "gold_standard.json")

    # Stage 2: Generate PnP-based pose network
    logger.info("=" * 20 + " STAGE 2: PnP Pose Network Generation " + "=" * 20)

    # Compute per-camera poses relative to object
    camera_to_object_poses = compute_camera_to_object_poses_pnp(image_points, camera_array)

    # Compute relative poses between camera pairs
    relative_poses = compute_relative_poses(camera_to_object_poses, camera_array)

    # Reject outliers per pair
    filtered_poses = reject_outliers(relative_poses)

    # Aggregate remaining poses
    aggregated_poses = aggregate_poses(filtered_poses)

    # Create PairedPoseNetwork with RMSE scores
    pnp_network = estimate_pnp_paired_pose_network(aggregated_poses, camera_array, image_points.df)
    save_network_to_json(pnp_network, output_dir / "pnp_estimates.json")

    # Stage 3: Validate against gold standard
    logger.info("=" * 20 + " STAGE 3: Gold Standard Comparison " + "=" * 20)
    compare_to_gold_standard(pnp_network, gold_standard_network, output_dir)

    logger.info("=" * 20 + " VALIDATION COMPLETE " + "=" * 20)
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
