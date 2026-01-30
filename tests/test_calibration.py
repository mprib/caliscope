import matplotlib

# Force non-interactive backend to prevent the debugger
# from trying to hook into the Qt GUI event loop.
matplotlib.use("Agg")


import logging
from pathlib import Path
from time import sleep
import numpy as np


from caliscope import __root__
from caliscope.core.bootstrap_pose.build_paired_pose_network import build_paired_pose_network


# from caliscope.cameras.camera_array_initializer import CameraArrayInitializer
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.core.point_data import ImagePoints
from caliscope.managers.synchronized_stream_manager import SynchronizedStreamManager
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope import persistence
from caliscope.core.point_data_bundle import PointDataBundle


logger = logging.getLogger(__name__)


def test_xy_charuco_creation(tmp_path: Path):
    original_session_path = Path(__root__, "tests", "sessions", "mediapipe_calibration")

    copy_contents_to_clean_dest(original_session_path, tmp_path)

    # This test begins with a set of cameras with calibrated intrinsics
    logger.info(f"Getting charuco from {tmp_path}")
    charuco = persistence.load_charuco(tmp_path / "charuco.toml")
    charuco_tracker = CharucoTracker(charuco)

    # create publishers for synchronized processing
    logger.info("Creating publishers")
    recording_path = Path(tmp_path, "calibration", "extrinsic")
    point_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")

    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    sync_stream_manager = SynchronizedStreamManager(
        recording_dir=recording_path, all_camera_data=camera_array.cameras, tracker=charuco_tracker
    )
    sync_stream_manager.process_streams(fps_target=100)

    # need to wait for points.csv file to populate
    while not point_data_path.exists():
        logger.info("Waiting for point_data.csv to populate...")
        sleep(1)

    assert point_data_path.exists()


def test_point_data_bundle_optimization(tmp_path: Path):
    version = "larger_calibration_post_monocal"
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    recording_path = Path(tmp_path, "calibration", "extrinsic")
    xy_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")

    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")

    image_points = ImagePoints.from_csv(xy_data_path)

    logger.info("Creating paired pose network")

    paired_pose_network = build_paired_pose_network(image_points, camera_array)
    logger.info("Initializing estimated camera positions based on best daisy-chained stereopairs")
    paired_pose_network.apply_to(camera_array, anchor_cam=8)

    # save initial extrinsics
    logger.info("Loading point estimates")
    image_points = ImagePoints.from_csv(xy_data_path)
    world_points = image_points.triangulate(camera_array)

    # Create initial bundle (triangulation)
    point_data_bundle = PointDataBundle(camera_array, image_points, world_points)

    initial_rmse = point_data_bundle.reprojection_report.overall_rmse
    logger.info(f"Initial RMSE (triangulation): {initial_rmse:.4f}px")
    logger.info(f"Initial observations: {len(point_data_bundle.image_points.df)}")

    # First optimization
    optimized_bundle = point_data_bundle.optimize()
    rmse_after_opt1 = optimized_bundle.reprojection_report.overall_rmse
    logger.info(f"RMSE after 1st optimization: {rmse_after_opt1:.4f}px")
    assert initial_rmse > rmse_after_opt1, "RMSE did not decline with first optimization"

    # Aggressive filtering (2px threshold) - should trigger safety mechanism
    filtered_bundle = optimized_bundle.filter_by_absolute_error(max_pixels=2.0, min_per_camera=50)
    rmse_after_filter = filtered_bundle.reprojection_report.overall_rmse
    logger.info(f"RMSE after filtering (2px): {rmse_after_filter:.4f}px")
    logger.info(f"Observations after filtering: {len(filtered_bundle.image_points.df)}")

    # Log per-camera observation counts to verify safety mechanism
    for port in sorted(filtered_bundle.camera_array.posed_cameras.keys()):
        count = (filtered_bundle.image_points.df["port"] == port).sum()
        logger.info(f"  Camera {port}: {count} observations")

    # RMSE should improve after filtering (removing worst outliers)
    assert rmse_after_opt1 > rmse_after_filter, "RMSE did not decline after filtering"

    # Second optimization on filtered data
    reoptimized_bundle = filtered_bundle.optimize()
    rmse_after_opt2 = reoptimized_bundle.reprojection_report.overall_rmse
    logger.info(f"RMSE after 2nd optimization: {rmse_after_opt2:.4f}px")

    # RMSE should improve further
    assert rmse_after_filter > rmse_after_opt2, "RMSE did not decline with second optimization"

    # Verify all cameras are still present
    posed_ports = set(reoptimized_bundle.camera_array.posed_cameras.keys())
    observed_ports = set(reoptimized_bundle.image_points.df["port"].unique())
    assert posed_ports == observed_ports, "Some cameras lost all observations!"

    logger.info("SUCCESS: RMSE decreased at each stage, all cameras retained")


def test_filter_percentile_modes(tmp_path: Path):
    """Verify that per_camera and overall percentile modes behave correctly and differently."""
    version = "larger_calibration_post_monocal"
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    recording_path = Path(tmp_path, "calibration", "extrinsic")
    xy_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")

    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    image_points = ImagePoints.from_csv(xy_data_path)
    world_points = image_points.triangulate(camera_array)

    bundle = PointDataBundle(camera_array, image_points, world_points)
    initial_rmse = bundle.reprojection_report.overall_rmse
    initial_obs = len(bundle.image_points.df)

    logger.info(f"Initial state: {initial_obs} observations, RMSE={initial_rmse:.4f}px")

    # Apply both modes with same percentile
    percentile = 95
    min_per_camera = 10

    bundle_per_cam = bundle.filter_by_percentile_error(
        percentile=percentile, scope="per_camera", min_per_camera=min_per_camera
    )
    bundle_overall = bundle.filter_by_percentile_error(
        percentile=percentile, scope="overall", min_per_camera=min_per_camera
    )

    n_per_cam = len(bundle_per_cam.image_points.df)
    n_overall = len(bundle_overall.image_points.df)
    logger.info(f"Per-camera: {n_per_cam} observations")
    logger.info(f"Overall: {n_overall} observations")

    # === ASSERTION 1: Both improve RMSE ===
    rmse_per_cam = bundle_per_cam.reprojection_report.overall_rmse
    rmse_overall = bundle_overall.reprojection_report.overall_rmse

    assert rmse_per_cam < initial_rmse, "Per-camera RMSE did not improve"
    assert rmse_overall < initial_rmse, "Overall RMSE did not improve"
    logger.info(f"RMSE improvement: {initial_rmse:.4f} → {rmse_per_cam:.4f} (per_camera)")
    logger.info(f"RMSE improvement: {initial_rmse:.4f} → {rmse_overall:.4f} (overall)")

    # === ASSERTION 2: Safety mechanism enforced ===
    for port in sorted(bundle.camera_array.posed_cameras.keys()):
        count_per_cam = (bundle_per_cam.image_points.df["port"] == port).sum()
        count_overall = (bundle_overall.image_points.df["port"] == port).sum()

        assert count_per_cam >= min_per_camera, f"Per-camera mode dropped camera {port} below minimum"
        assert count_overall >= min_per_camera, f"Overall mode dropped camera {port} below minimum"

        logger.info(f"  Camera {port}: {count_per_cam} (per_cam), {count_overall} (overall)")

    # === ASSERTION 3: No orphaned 3D points ===
    # Every 3D point should have at least one observation
    per_cam_points = set(zip(bundle_per_cam.world_points.df["sync_index"], bundle_per_cam.world_points.df["point_id"]))
    per_cam_obs_keys = set(
        zip(bundle_per_cam.image_points.df["sync_index"], bundle_per_cam.image_points.df["point_id"])
    )
    assert per_cam_points.issubset(per_cam_obs_keys), "Per-camera has orphaned 3D points!"

    overall_points = set(zip(bundle_overall.world_points.df["sync_index"], bundle_overall.world_points.df["point_id"]))
    overall_obs_keys = set(
        zip(bundle_overall.image_points.df["sync_index"], bundle_overall.image_points.df["point_id"])
    )
    assert overall_points.issubset(overall_obs_keys), "Overall has orphaned 3D points!"

    # === ASSERTION 4: Percentile math is roughly correct ===
    # For overall mode, should remove ~percentile% of observations
    expected_removal = initial_obs * (percentile / 100)
    actual_removal = initial_obs - n_overall
    removal_ratio = actual_removal / expected_removal

    # Allow 20% tolerance due to safety mechanism
    assert 0.8 <= removal_ratio <= 1.2, f"Overall mode removed {actual_removal} obs, expected ~{expected_removal}"
    logger.info(f"Overall mode removed {actual_removal} of {expected_removal:.0f} expected observations")

    # === ASSERTION 5: Per-camera mode respects camera error distributions ===
    # Cameras with higher initial RMSE should lose more observations in per_camera mode
    initial_by_camera = bundle.reprojection_report.by_camera

    # Calculate removal fraction per camera for per_camera mode
    removal_fractions = {}
    for port in initial_by_camera.keys():
        initial_count = (bundle.image_points.df["port"] == port).sum()
        final_count = (bundle_per_cam.image_points.df["port"] == port).sum()
        removal_fractions[port] = (initial_count - final_count) / initial_count

    # Sort cameras by initial RMSE
    sorted_by_rmse = sorted(initial_by_camera.items(), key=lambda x: x[1])

    # Check that higher-RMSE cameras generally have higher removal fractions
    # (This is a statistical tendency, not a strict guarantee)
    high_rmse_ports = [port for port, rmse in sorted_by_rmse[-3:]]  # Top 3 worst cameras
    low_rmse_ports = [port for port, rmse in sorted_by_rmse[:3]]  # Top 3 best cameras

    avg_removal_high = np.mean([removal_fractions[p] for p in high_rmse_ports])
    avg_removal_low = np.mean([removal_fractions[p] for p in low_rmse_ports])

    logger.info(f"Avg removal fraction - high RMSE cameras: {avg_removal_high:.2%}")
    logger.info(f"Avg removal fraction - low RMSE cameras: {avg_removal_low:.2%}")

    # High RMSE cameras should have higher removal fraction (allow 5% tolerance)
    assert avg_removal_high > avg_removal_low - 0.05, "Per-camera mode didn't target high-error cameras!"

    # === ASSERTION 6: Overall mode also targets high-error cameras ===
    # Calculate removal fraction per camera for overall mode
    removal_fractions_overall = {}
    for port in initial_by_camera.keys():
        initial_count = (bundle.image_points.df["port"] == port).sum()
        final_count = (bundle_overall.image_points.df["port"] == port).sum()
        removal_fractions_overall[port] = (initial_count - final_count) / initial_count

    avg_removal_high_overall = np.mean([removal_fractions_overall[p] for p in high_rmse_ports])
    avg_removal_low_overall = np.mean([removal_fractions_overall[p] for p in low_rmse_ports])

    logger.info(f"Avg removal fraction (overall) - high RMSE cameras: {avg_removal_high_overall:.2%}")
    logger.info(f"Avg removal fraction (overall) - low RMSE cameras: {avg_removal_low_overall:.2%}")

    # Even with global threshold, high RMSE cameras should lose more
    assert avg_removal_high_overall > avg_removal_low_overall - 0.05, "Overall mode didn't target high-error cameras!"

    logger.info("✓ All percentile mode assertions passed")


if __name__ == "__main__":
    from caliscope.logger import setup_logging

    setup_logging()

    temp_path = Path(__file__).parent / "debug"
    test_point_data_bundle_optimization(temp_path)
    test_xy_charuco_creation(temp_path)
    test_filter_percentile_modes(temp_path)
