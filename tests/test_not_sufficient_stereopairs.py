""" """

import logging
from pathlib import Path


from caliscope import __root__
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.capture_volume.quality_controller import QualityController
from caliscope.calibration.bootstrap_pose.build_paired_pose_network import build_paired_pose_network
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.post_processing.point_data import ImagePoints
from caliscope import persistence

logger = logging.getLogger(__name__)

# don't want ruff dropping the reference which I use in repl


def test_calibration_workflow(tmp_path: Path):
    version = "not_sufficient_stereopairs"
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    xy_data_path = Path(tmp_path, "xy_CHARUCO.csv")
    #    camera_array = config.get_camera_array()
    charuco = persistence.load_charuco(tmp_path / "charuco.toml")
    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    logger.info("Creating stereocalibrator")

    image_points = ImagePoints.from_csv(xy_data_path)

    logger.info("Initiating stereocalibration")
    paired_pose_network = build_paired_pose_network(image_points, camera_array)

    logger.info("Initializing estimated camera positions based on best daisy-chained stereopairs")
    paired_pose_network.apply_to(camera_array)

    logger.info("Loading point estimates")
    world_points = image_points.triangulate(camera_array)
    point_estimates: PointEstimates = world_points.to_point_estimates(image_points, camera_array)

    capture_volume = CaptureVolume(camera_array, point_estimates)

    # Before filtering - log initial point counts
    logger.info("========== POINT COUNT DIAGNOSTICS ==========")
    logger.info("Initial point counts:")
    logger.info(f"  3D points (obj.shape[0]): {capture_volume.point_estimates.obj.shape[0]}")
    logger.info(f"  2D observations (img.shape[0]): {capture_volume.point_estimates.img.shape[0]}")
    logger.info(f"  Camera indices length: {len(capture_volume.point_estimates.camera_indices)}")

    QualityController(capture_volume, charuco)

    # Verify initial state
    assert capture_volume.stage == 0
    rmse_initial = capture_volume.rmse
    assert rmse_initial is not None
    assert "overall" in rmse_initial
    assert all(str(port) in rmse_initial for port in capture_volume.camera_array.posed_cameras.keys())

    # Log initial RMSE values
    logger.info(f"Initial RMSE before optimization: {rmse_initial['overall']:.4f} pixels")
    logger.info("Per-camera initial RMSE values:")

    # First optimization stage - bundle adjustment
    logger.info("Performing bundle adjustment")
    capture_volume.optimize(ftol=1e-3)
    assert capture_volume.stage == 1

    # Log post-bundle adjustment RMSE and improvement
    rmse_post_bundle_adj = capture_volume.rmse
    improvement = rmse_initial["overall"] - rmse_post_bundle_adj["overall"]
    percent_improvement = (improvement / rmse_initial["overall"]) * 100
    logger.info(f"RMSE after bundle adjustment: {rmse_post_bundle_adj['overall']:.4f} pixels")
    logger.info(f"Improvement: {improvement:.4f} pixels ({percent_improvement:.2f}%)")
    assert rmse_post_bundle_adj["overall"] <= rmse_initial["overall"]


if __name__ == "__main__":
    from caliscope.logger import setup_logging

    setup_logging()
    temp_path = Path(__file__).parent / "debug"
    test_calibration_workflow(temp_path)
