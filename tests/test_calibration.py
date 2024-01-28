import caliscope.logger

from time import sleep
import shutil
from pathlib import Path
from caliscope.cameras.camera_array import CameraArray
from caliscope import __root__
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.cameras.camera_array_initializer import CameraArrayInitializer
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
import pytest
from caliscope.trackers.charuco_tracker import CharucoTracker

from caliscope.calibration.stereocalibrator import StereoCalibrator
from caliscope.calibration.capture_volume.quality_controller import QualityController

from caliscope.synchronized_stream_manager import SynchronizedStreamManager


from caliscope.controller import FILTERED_FRACTION
from caliscope.configurator import Configurator

logger = caliscope.logger.get(__name__)

TEST_SESSIONS = ["mediapipe_calibration"]


def copy_contents(src_folder, dst_folder):
    """
    Helper function to port a test case data folder over to a temp directory
    used for testing purposes so that the test case data doesn't get overwritten
    """
    src_path = Path(src_folder)
    dst_path = Path(dst_folder)

    # Create the destination folder if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=True)

    for item in src_path.iterdir():
        # Construct the source and destination paths
        src_item = src_path / item
        dst_item = dst_path / item.name

        # Copy file or directory
        if src_item.is_file():
            logger.info(f"Copying file at {src_item} to {dst_item}")
            shutil.copy2(src_item, dst_item)  # Copy file preserving metadata

        elif src_item.is_dir():
            logger.info(f"Copying directory at {src_item} to {dst_item}")
            shutil.copytree(src_item, dst_item)


@pytest.fixture(params=TEST_SESSIONS)
def session_path(request, tmp_path):
    """
    Tests will be run based on data stored in tests/sessions, but to avoid overwriting
    or altering test cases,the tested directory will get copied over to a temp
    directory and then that temp directory will be passed on to the calling functions
    """
    original_test_data_path = Path(__root__, "tests", "sessions", request.param)
    tmp_test_data_path = Path(tmp_path, request.param)
    copy_contents(original_test_data_path, tmp_test_data_path)

    return tmp_test_data_path
    # return original_test_data_path


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
    point_data_path = Path(recording_path,"CHARUCO", "xy_CHARUCO.csv")

    camera_array = config.get_camera_array()
    sync_stream_manager = SynchronizedStreamManager(
        recording_dir=recording_path, all_camera_data= camera_array.cameras, tracker=charuco_tracker
    )
    sync_stream_manager.process_streams(fps_target=100)

    # need to wait for points.csv file to populate
    while not point_data_path.exists():
        logger.info("Waiting for point_data.csv to populate...")
        sleep(1)

    logger.info("Waiting for video recorder to finish processing stream...")
    stereocalibrator = StereoCalibrator(config.config_toml_path, point_data_path)
    stereocalibrator.stereo_calibrate_all(boards_sampled=10)

    camera_array: CameraArray = CameraArrayInitializer(
        config.config_toml_path
    ).get_best_camera_array()

    point_estimates: PointEstimates = get_point_estimates(camera_array, point_data_path)

    capture_volume = CaptureVolume(camera_array, point_estimates)
    initial_rmse = capture_volume.rmse
    logger.info(f"Prior to bundle adjustment, RMSE error is {initial_rmse}")
    capture_volume.optimize()

    quality_controller = QualityController(capture_volume, charuco)
    # Removing the worst fitting {FILTERED_FRACTION*100} percent of points from the model
    logger.info(f"Filtering out worse fitting {FILTERED_FRACTION*100} % of points")
    quality_controller.filter_point_estimates(FILTERED_FRACTION)
    logger.info("Re-optimizing with filtered data set")
    capture_volume.optimize()
    optimized_filtered_rmse = capture_volume.rmse

    # save out results of optimization for later assessment with F5 test walkthroughs
    config.save_camera_array(capture_volume.camera_array)
    config.save_point_estimates(capture_volume.point_estimates)

    for key, optimized_rmse in optimized_filtered_rmse.items():
        logger.info(f"Asserting that RMSE decreased with optimization at {key}...")
        assert initial_rmse[key] > optimized_rmse


if __name__ == "__main__":
    original_session_path = Path(__root__, "tests", "sessions", "mediapipe_calibration")
    session_path = Path(
        original_session_path.parent.parent,
        "sessions_copy_delete",
        "mediapipe_calibration",
    )

    # clear previous test so as not to pollute current test results
    if session_path.exists() and session_path.is_dir():
        shutil.rmtree(session_path)

    copy_contents(original_session_path, session_path)

    test_post_monocalibration(session_path)
