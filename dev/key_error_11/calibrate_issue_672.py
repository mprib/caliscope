import shutil
from pathlib import Path
from time import sleep


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
from caliscope.synchronized_stream_manager import SynchronizedStreamManager
from caliscope.trackers.charuco_tracker import CharucoTracker

logger = caliscope.logger.get(__name__)

TEST_SESSIONS = ["mediapipe_calibration"]


def test_post_monocalibration(session_path):
    # This test begins with a set of cameras with calibrated intrinsics
    config = Configurator(session_path)
    # config_path = str(Path(session_path, "config.toml"))
    logger.info(f"Getting charuco from config at {config.config_toml_path}")
    charuco = config.get_charuco()
    charuco_tracker = CharucoTracker(charuco)

    # create a synchronizer based off of these stream pools
    logger.info("Creating RecordedStreamPool")

    recording_path = Path(session_path, "calibration", "extrinsic")
    point_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")

    camera_array = config.get_camera_array()
    # sync_stream_manager = SynchronizedStreamManager(
    #     recording_dir=recording_path, all_camera_data=camera_array.cameras, tracker=charuco_tracker
    # )
    # sync_stream_manager.process_streams(fps_target=100)

    # # need to wait for points.csv file to populate
    # while not point_data_path.exists():
    #     logger.info("Waiting for point_data.csv to populate...")
    #     sleep(1)

    logger.info("Waiting for video recorder to finish processing stream...")
    stereocalibrator = StereoCalibrator(config.config_toml_path, point_data_path)
    stereocalibrator.stereo_calibrate_all(boards_sampled=5)

    camera_array: CameraArray = CameraArrayInitializer(config.config_toml_path).get_best_camera_array()

    point_estimates: PointEstimates = get_point_estimates(camera_array, point_data_path)

    capture_volume = CaptureVolume(camera_array, point_estimates)
    initial_rmse = capture_volume.rmse
    logger.info(f"Prior to bundle adjustment, RMSE error is {initial_rmse}")
    capture_volume.optimize()

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
    session_path = Path(__root__, "dev", "key_error_11")

    # clear previous test so as not to pollute current test results
    test_post_monocalibration(session_path)
