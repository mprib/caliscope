""" """

import logging
from pathlib import Path


from caliscope import __root__
from caliscope.core.bootstrap_pose.build_paired_pose_network import build_paired_pose_network
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.core.point_data import ImagePoints
from caliscope.core.point_data_bundle import PointDataBundle
from caliscope import persistence

logger = logging.getLogger(__name__)

# don't want ruff dropping the reference which I use in repl


def test_calibration_workflow(tmp_path: Path):
    version = "not_sufficient_stereopairs"
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    xy_data_path = Path(tmp_path, "xy_CHARUCO.csv")
    #    camera_array = config.get_camera_array()
    persistence.load_charuco(tmp_path / "charuco.toml")
    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    logger.info("Creating stereocalibrator")

    image_points = ImagePoints.from_csv(xy_data_path)

    logger.info("Initiating stereocalibration")
    paired_pose_network = build_paired_pose_network(image_points, camera_array)

    logger.info("Initializing estimated camera positions based on best daisy-chained stereopairs")
    paired_pose_network.apply_to(camera_array)

    logger.info("Triangulating world points")
    world_points = image_points.triangulate(camera_array)

    # Create initial bundle
    bundle = PointDataBundle(camera_array, image_points, world_points)

    # Before filtering - log initial point counts
    logger.info("========== POINT COUNT DIAGNOSTICS ==========")
    logger.info("Initial point counts:")
    logger.info(f"  3D points: {len(bundle.world_points.df)}")
    logger.info(f"  2D observations: {len(bundle.image_points.df)}")
    logger.info(f"  Cameras: {len(bundle.camera_array.posed_cameras)}")

    # Verify initial state
    rmse_initial = bundle.reprojection_report.overall_rmse
    assert rmse_initial > 0
    per_camera_rmse = bundle.reprojection_report.by_camera
    assert all(port in per_camera_rmse for port in bundle.camera_array.posed_cameras.keys())

    # Log initial RMSE values
    logger.info(f"Initial RMSE before optimization: {rmse_initial:.4f} pixels")
    logger.info("Per-camera initial RMSE values:")
    for port, rmse in sorted(per_camera_rmse.items()):
        logger.info(f"  Camera {port}: {rmse:.4f} pixels")

    # First optimization stage - bundle adjustment
    logger.info("Performing bundle adjustment")
    optimized_bundle = bundle.optimize(ftol=1e-3)

    # Log post-bundle adjustment RMSE and improvement
    rmse_post_bundle_adj = optimized_bundle.reprojection_report.overall_rmse
    improvement = rmse_initial - rmse_post_bundle_adj
    percent_improvement = (improvement / rmse_initial) * 100
    logger.info(f"RMSE after bundle adjustment: {rmse_post_bundle_adj:.4f} pixels")
    logger.info(f"Improvement: {improvement:.4f} pixels ({percent_improvement:.2f}%)")
    assert rmse_post_bundle_adj <= rmse_initial


if __name__ == "__main__":
    from caliscope.logger import setup_logging

    setup_logging()
    temp_path = Path(__file__).parent / "debug"
    test_calibration_workflow(temp_path)
