""" """

from pathlib import Path

import numpy as np

import caliscope.logger
from caliscope import __root__
from caliscope.calibration.stereocalibrator import StereoCalibrator
from caliscope.cameras.camera_array import CameraArray
from caliscope.cameras.camera_array_initializer import CameraArrayInitializer
from caliscope.configurator import Configurator
from caliscope.helper import copy_contents

logger = caliscope.logger.get(__name__)


def test_missing_extrinsics_old():
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
    config.get_camera_array()
    config.get_charuco()

    logger.info("Creating stereocalibrator")
    stereocalibrator = StereoCalibrator(config.config_toml_path, xy_data_path)
    logger.info("Initiating stereocalibration")
    stereocalibrator.stereo_calibrate_all(boards_sampled=10)

    logger.info("stereocalibration complete")
    logger.info("Initializing estimated camera positions based on best daisy-chained stereopairs")
    camera_array: CameraArray = CameraArrayInitializer(config.config_toml_path).get_best_camera_array()
    logger.info("Camera Poses estimated from stereocalibration")
    # should have posed all ports but 5
    assert list(camera_array.posed_cameras.keys()) == [1, 2, 3, 4, 6]
    assert list(camera_array.unposed_cameras.keys()) == [5]

    # when creating extrinsic parameters shouldn't have camera 5.
    # should be able to extract params from complete extrinsics vector and map to individual cam params


def test_missing_extrinsics():
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
    config.get_camera_array()
    config.get_charuco()

    logger.info("Creating stereocalibrator")
    stereocalibrator = StereoCalibrator(config.config_toml_path, xy_data_path)
    logger.info("Initiating stereocalibration")
    stereocalibrator.stereo_calibrate_all(boards_sampled=10)

    logger.info("stereocalibration complete")
    logger.info("Initializing estimated camera positions based on best daisy-chained stereopairs")
    camera_array: CameraArray = CameraArrayInitializer(config.config_toml_path).get_best_camera_array()
    logger.info("Camera Poses estimated from stereocalibration")

    # should have posed all ports but 5
    # Using set for posed_cameras to avoid order dependency
    assert set(camera_array.posed_cameras.keys()) == {1, 2, 3, 4, 6}
    assert list(camera_array.unposed_cameras.keys()) == [5]

    # when creating extrinsic parameters shouldn't have camera 5.
    extrinsic_params = camera_array.get_extrinsic_params()
    assert extrinsic_params is not None, "Extrinsic parameters should not be None"
    assert extrinsic_params.shape == (5, 6), "Shape should be (5 posed cameras, 6 params)"

    # camera 5 should not be in the index used for optimization parameter mapping
    assert 5 not in camera_array.posed_port_to_index

    # Verify that the order of cameras in the extrinsic_params array is correct
    logger.info("Verifying order of extrinsic parameters vector...")
    for port, index in camera_array.posed_port_to_index.items():
        expected_params = camera_array.cameras[port].extrinsics_to_vector()
        actual_params = extrinsic_params[index]
        np.testing.assert_array_equal(
            actual_params, expected_params, err_msg=f"Parameter mismatch for port {port} at index {index}"
        )

    # should be able to extract params from complete extrinsics vector and map back to individual cam params
    # This round-trip test confirms the mapping from vector -> cameras works correctly

    # 1. Simulate a small change from an optimization step
    new_params = extrinsic_params + 0.01

    # 2. Update the camera array with the new parameters
    camera_array.update_extrinsic_params(new_params)

    # 3. Verify the update worked correctly on the posed cameras
    updated_params = camera_array.get_extrinsic_params()
    assert updated_params is not None
    np.testing.assert_allclose(updated_params, new_params, atol=1e-6)

    # 4. Verify the unposed camera was untouched
    unposed_cam = camera_array.cameras[5]
    assert unposed_cam.rotation is None, "Unposed camera rotation should remain None"
    assert unposed_cam.translation is None, "Unposed camera translation should remain None"


if __name__ == "__main__":
    test_missing_extrinsics()
    # test_deterministic_consistency()
