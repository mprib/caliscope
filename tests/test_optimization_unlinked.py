from pathlib import Path

import caliscope.logger
from caliscope import __root__
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.calibration.capture_volume.helper_functions.get_point_estimates import (
    create_point_estimates_from_stereopairs,
)
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.capture_volume.quality_controller import QualityController
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

    camera_array = config.get_camera_array()
    config.get_charuco()

    logger.info("Creating stereocalibrator")
    stereocalibrator = StereoCalibrator(camera_array, xy_data_path)

    logger.info("Initiating stereocalibration")
    stereo_results = stereocalibrator.stereo_calibrate_all()

    # 2. INITIALIZE CAMERA_ARRAY
    # This will result in camera 5 being unposed
    initializer = CameraArrayInitializer(camera_array, stereo_results)

    camera_array: CameraArray = initializer.get_best_camera_array()

    # 3. VERIFY SETUP
    # Confirm that we have the expected set of posed and unposed cameras
    # (cam 5 no shared images and cam 4 actively ignored)
    assert set(camera_array.posed_cameras.keys()) == {1, 2, 3, 6}
    assert list(camera_array.unposed_cameras.keys()) == [4, 5]
    assert len(camera_array.posed_port_to_index) == 4  # Critical: only 4 cameras are indexed for optimization

    # 4. GENERATE POINT ESTIMATES
    # This is the function we will modify. Currently, it will fail to correctly
    # filter and remap camera indices.
    logger.info("Generating point estimates from data with unlinked camera...")
    point_estimates: PointEstimates = create_point_estimates_from_stereopairs(camera_array, xy_data_path)

    # 5. CREATE CAPTURE VOLUME AND OPTIMIZE
    # This step will fail with an IndexError before our changes, because
    # PointEstimates contains camera indices that are out of bounds for the
    # optimization parameter array.
    logger.info("Creating CaptureVolume and running optimization...")
    capture_volume = CaptureVolume(camera_array, point_estimates)
    logger.info(f"Initial rmse: {capture_volume.get_rmse_summary()}")
    # The core of the test: can it optimize without crashing?
    capture_volume.optimize()

    # saving to create a new test
    config.save_capture_volume(capture_volume)
    logger.info(f"saving capture volume to {config.config_toml_path.parent}")

    # 6. ASSERT SUCCESS
    # If optimize() completes, the test has passed.
    assert capture_volume.stage == 1
    logger.info("Optimization completed successfully with an unlinked camera present.")


def test_capture_volume_filter():
    # 1. SETUP: Use output of test_bundle_adjust_with_unlinked_camera  as starting point
    version = "capture_volume_pre_quality_control"
    original_session_path = Path(__root__, "tests", "sessions", version)
    session_path = Path(
        original_session_path.parent.parent,
        "sessions_copy_delete",
        version,
    )
    copy_contents(original_session_path, session_path)

    config = Configurator(session_path)
    camera_array: CameraArray = config.get_camera_array()
    point_estimates: PointEstimates = config.load_point_estimates_from_toml()

    logger.info("Camera array and point estimates loaded... creating capture volume")
    capture_volume = CaptureVolume(camera_array, point_estimates)
    logger.info("CaptureVolume initialized")

    capture_volume._save(directory=session_path, descriptor="initial")
    capture_volume.optimize()
    capture_volume._save(directory=session_path, descriptor="post_optimization")

    logger.info("Point counts BEFORE filtering:")
    logger.info(f"  3D points (obj.shape[0]): {capture_volume.point_estimates.obj.shape[0]}")
    logger.info(f"  2D observations (img.shape[0]): {capture_volume.point_estimates.img.shape[0]}")
    logger.info(f"  Camera indices length: {len(capture_volume.point_estimates.camera_indices)}")

    filtered_fraction = 0.5
    logger.info(f"Filtering out worse fitting {filtered_fraction * 100:.1f}% of points")
    charuco = config.get_charuco()
    quality_controller = QualityController(capture_volume, charuco)
    quality_controller.filter_point_estimates(filtered_fraction)
    capture_volume._save(directory=session_path, descriptor="post_filtering")

    logger.info("Point counts AFTER filtering:")
    logger.info(f"  3D points (obj.shape[0]): {capture_volume.point_estimates.obj.shape[0]}")
    logger.info(f"  2D observations (img.shape[0]): {capture_volume.point_estimates.img.shape[0]}")
    logger.info(f"  Camera indices length: {len(capture_volume.point_estimates.camera_indices)}")

    capture_volume.optimize()
    capture_volume._save(directory=session_path, descriptor="post_filtering_then_optimizing")


if __name__ == "__main__":
    # test_bundle_adjust_with_unlinked_camera()
    test_capture_volume_filter()
    logger.info("test debug complete")
