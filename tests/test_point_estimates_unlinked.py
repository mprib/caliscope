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

logger = caliscope.logger.get(__name__)


def test_bundle_adjust_with_unlinked_camera():
    """
    Tests the full pipeline from initializing a CameraArray with a missing
    camera, through point estimate generation, to bundle adjustment.
    This ensures that PointEstimates are correctly filtered and re-indexed
    to match the posed cameras in the CameraArray.
    """
    # 1. SETUP: Use test data that results in an unposed camera
    version = "not_sufficient_stereopairs"
    original_session_path = Path(__root__, "tests", "sessions", version)
    session_path = Path(
        original_session_path.parent.parent,
        "sessions_copy_delete",
        version,
    )
    copy_contents(original_session_path, session_path)

    config = Configurator(session_path)
    # The xy_CHARUCO.csv is at the root of the session for this test case
    xy_data_path = Path(session_path, "xy_CHARUCO.csv")

    config.get_camera_array()
    config.get_charuco()

    logger.info("Creating stereocalibrator")
    stereocalibrator = StereoCalibrator(config.config_toml_path, xy_data_path)
    logger.info("Initiating stereocalibration")

    stereocalibrator.stereo_calibrate_all(boards_sampled=10)
    # 2. INITIALIZE CAMERA_ARRAY
    # This will result in camera 5 being unposed
    camera_array: CameraArray = CameraArrayInitializer(config.config_toml_path).get_best_camera_array()

    # 3. VERIFY SETUP
    # Confirm that we have the expected set of posed and unposed cameras
    assert set(camera_array.posed_cameras.keys()) == {1, 2, 3, 4, 6}
    assert list(camera_array.unposed_cameras.keys()) == [5]
    assert len(camera_array.posed_port_to_index) == 5  # Critical: only 5 cameras are indexed for optimization

    # 4. GENERATE POINT ESTIMATES
    # This is the function we will modify. Currently, it will fail to correctly
    # filter and remap camera indices.
    logger.info("Generating point estimates from data with unlinked camera...")
    point_estimates: PointEstimates = get_point_estimates(camera_array, xy_data_path)

    # 5. CREATE CAPTURE VOLUME AND OPTIMIZE
    # This step will fail with an IndexError before our changes, because
    # PointEstimates contains camera indices that are out of bounds for the
    # optimization parameter array.
    logger.info("Creating CaptureVolume and running optimization...")
    capture_volume = CaptureVolume(camera_array, point_estimates)

    # The core of the test: can it optimize without crashing?
    capture_volume.optimize()

    # 6. ASSERT SUCCESS
    # If optimize() completes, the test has passed.
    assert capture_volume.stage == 1
    logger.info("Optimization completed successfully with an unlinked camera present.")


if __name__ == "__main__":
    test_bundle_adjust_with_unlinked_camera()
