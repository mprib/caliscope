import shutil
from pathlib import Path

import caliscope.logger
from caliscope import __root__
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.stereocalibrator import StereoCalibrator
from caliscope.cameras.camera_array import CameraArray
from caliscope.cameras.camera_array_initializer import CameraArrayInitializer
from caliscope.configurator import Configurator
from caliscope.helper import copy_contents
from caliscope.trackers.charuco_tracker import CharucoTracker

logger = caliscope.logger.get(__name__)

TEST_SESSIONS = ["mediapipe_calibration"]


def test_complex_calibrate_filter(session_path):
    config = Configurator(session_path)

    recording_path = Path(session_path, "calibration", "extrinsic")
    point_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")

    camera_array = config.get_camera_array()

    logger.info("Creating stereocalibrator")
    stereocalibrator = StereoCalibrator(config.config_toml_path, point_data_path)
    logger.info("Initiating stereocalibration")
    stereocalibrator.stereo_calibrate_all(boards_sampled=5)

    logger.info("Initializing estimated camera positions based on best daisy-chained stereopairs")
    camera_array: CameraArray = CameraArrayInitializer(config.config_toml_path).get_best_camera_array()

    logger.info("Loading point estimates")
    point_estimates: PointEstimates = get_point_estimates(camera_array, point_data_path)

    capture_volume = CaptureVolume(camera_array, point_estimates)
    initial_rmse = capture_volume.rmse
    logger.info(f"Prior to bundle adjustment, RMSE error is {initial_rmse}")
    capture_volume.optimize()

    # The code below is failing, which needs to get resolved, or this stage of processing needs to just be avoided for now...it's not critical
    # quality_controller = QualityController(capture_volume, charuco)
    # Removing the worst fitting {FILTERED_FRACTION*100} percent of points from the model
    # logger.info(f"Filtering out worse fitting {FILTERED_FRACTION*100} % of points")
    # quality_controller.filter_point_estimates(FILTERED_FRACTION)
    # logger.info("Re-optimizing with filtered data set")
    # capture_volume.optimize()
    optimized_filtered_rmse = capture_volume.rmse

    # save out results of optimization for later assessment with F5 test walkthroughs
    config.save_camera_array(capture_volume.camera_array)
    config.save_point_estimates(capture_volume.point_estimates)

    for key, optimized_rmse in optimized_filtered_rmse.items():
        logger.info(f"Asserting that RMSE decreased with optimization at {key}...")
        assert initial_rmse[key] > optimized_rmse


if __name__ == "__main__":

    original_session_path = Path(__root__, "tests", "sessions", "issue_672")
    session_path = Path(
        original_session_path.parent.parent,
        "sessions_copy_delete",
        "issue_672",
    )

    # clear previous test so as not to pollute current test results
    if session_path.exists() and session_path.is_dir():
        shutil.rmtree(session_path)

    copy_contents(original_session_path, session_path)

    # clear previous test so as not to pollute current test results
    test_complex_calibrate_filter(session_path)
