"""
Attempting to duplicate the error in Issue #690.
The issue stems from camera 5 not having common board views with any of the other cameras.
The test below runs no assertions and is just a starting point for future reference
to build out a more robust functionality.
"""

from pathlib import Path

import caliscope.logger
from caliscope import __root__
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.capture_volume.quality_controller import QualityController
from caliscope.calibration.stereocalibrator import StereoCalibrator
from caliscope.cameras.camera_array import CameraArray
from caliscope.cameras.camera_array_initializer import CameraArrayInitializer
from caliscope.configurator import Configurator
from caliscope.controller import FILTERED_FRACTION
from caliscope.helper import copy_contents

logger = caliscope.logger.get(__name__)


def test_calibration():
    version = "not_sufficient_stereopairs"
    original_session_path = Path(__root__, "tests", "sessions", version)
    session_path = Path(
        original_session_path.parent.parent,
        "sessions_copy_delete",
        version,
    )
    copy_contents(original_session_path, session_path)

    config = Configurator(session_path)
    xy_data_path = Path(session_path, "xy_CHARUCO.csv")
    camera_array = config.get_camera_array()
    charuco = config.get_charuco()

    logger.info("Creating stereocalibrator")
    stereocalibrator = StereoCalibrator(config.config_toml_path, xy_data_path)
    logger.info("Initiating stereocalibration")
    stereocalibrator.stereo_calibrate_all(boards_sampled=100)

    logger.info("Initializing estimated camera positions based on best daisy-chained stereopairs")
    camera_array: CameraArray = CameraArrayInitializer(config.config_toml_path).get_best_camera_array()

    logger.info("Loading point estimates")
    point_estimates: PointEstimates = get_point_estimates(camera_array, xy_data_path)

    capture_volume = CaptureVolume(camera_array, point_estimates)

    # Before filtering - log initial point counts
    logger.info("========== POINT COUNT DIAGNOSTICS ==========")
    logger.info("Initial point counts:")
    logger.info(f"  3D points (obj.shape[0]): {capture_volume.point_estimates.obj.shape[0]}")
    logger.info(f"  2D observations (img.shape[0]): {capture_volume.point_estimates.img.shape[0]}")
    logger.info(f"  Camera indices length: {len(capture_volume.point_estimates.camera_indices)}")
    logger.info(f"  Saving to path: {config.point_estimates_toml_path}")

    quality_controller = QualityController(capture_volume, charuco)

    # Verify initial state
    assert capture_volume.stage == 0
    rmse_initial = capture_volume.rmse
    assert rmse_initial is not None
    assert "overall" in rmse_initial
    assert all(str(port) in rmse_initial for port in capture_volume.camera_array.cameras.keys())

    # Log initial RMSE values
    logger.info(f"Initial RMSE before optimization: {rmse_initial['overall']:.4f} pixels")
    logger.info("Per-camera initial RMSE values:")
    for port in capture_volume.camera_array.cameras.keys():
        logger.info(f"  Camera {port}: {rmse_initial[str(port)]:.4f} pixels")

    # First optimization stage - bundle adjustment
    logger.info("Performing bundle adjustment")
    capture_volume.optimize()
    assert capture_volume.stage == 1

    # Log post-bundle adjustment RMSE and improvement
    rmse_post_bundle_adj = capture_volume.rmse
    improvement = rmse_initial["overall"] - rmse_post_bundle_adj["overall"]
    percent_improvement = (improvement / rmse_initial["overall"]) * 100
    logger.info(f"RMSE after bundle adjustment: {rmse_post_bundle_adj['overall']:.4f} pixels")
    logger.info(f"Improvement: {improvement:.4f} pixels ({percent_improvement:.2f}%)")
    assert rmse_post_bundle_adj["overall"] <= rmse_initial["overall"]

    # Second stage - filter out worse points
    logger.info(f"Filtering out worse fitting {FILTERED_FRACTION * 100:.1f}% of points")
    quality_controller.filter_point_estimates(FILTERED_FRACTION)

    # After filtering - log filtered point counts
    logger.info("Point counts AFTER filtering:")
    logger.info(f"  3D points (obj.shape[0]): {capture_volume.point_estimates.obj.shape[0]}")
    logger.info(f"  2D observations (img.shape[0]): {capture_volume.point_estimates.img.shape[0]}")
    logger.info(f"  Camera indices length: {len(capture_volume.point_estimates.camera_indices)}")

    # Log post-filtering RMSE (before re-optimization)
    rmse_post_filter = capture_volume.rmse
    logger.info(f"RMSE after filtering (before re-optimization): {rmse_post_filter['overall']:.4f} pixels")

    # Final stage - re-optimize with filtered data
    logger.info("Re-optimizing with filtered data set")
    capture_volume.optimize()

    # Log final RMSE and total improvement
    rmse_final = capture_volume.rmse
    total_improvement = rmse_initial["overall"] - rmse_final["overall"]
    total_percent = (total_improvement / rmse_initial["overall"]) * 100
    filter_improvement = rmse_post_bundle_adj["overall"] - rmse_final["overall"]
    filter_percent = (filter_improvement / rmse_post_bundle_adj["overall"]) * 100

    logger.info(f"Final RMSE after filtering and re-optimization: {rmse_final['overall']:.4f} pixels")
    logger.info(f"Improvement from filtering: {filter_improvement:.4f} pixels ({filter_percent:.2f}%)")
    logger.info(f"Total improvement: {total_improvement:.4f} pixels ({total_percent:.2f}%)")
    logger.info("Per-camera final RMSE values:")
    for port in capture_volume.camera_array.cameras.keys():
        initial = rmse_initial[str(port)]
        final = rmse_final[str(port)]
        cam_improvement = (initial - final) / initial * 100
        logger.info(f"  Camera {port}: {final:.4f} pixels (improved {cam_improvement:.2f}%)")

    config.save_point_estimates(capture_volume.point_estimates)
    config.save_camera_array(capture_volume.camera_array)
    config.save_capture_volume(capture_volume)


if __name__ == "__main__":
    test_calibration()
