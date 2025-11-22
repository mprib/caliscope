""" """

import logging
from pathlib import Path

import numpy as np

from caliscope import __root__
from caliscope.calibration.stereocalibrator import StereoCalibrator
from caliscope.cameras.camera_array import CameraArray
from caliscope.cameras.camera_array_initializer import CameraArrayInitializer
from caliscope.configurator import Configurator
from caliscope.helper import copy_contents
from caliscope.post_processing.point_data import ImagePoints

logger = logging.getLogger(__name__)


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
    camera_array = config.get_camera_array()
    logger.info("Creating stereocalibrator")

    image_points = ImagePoints.from_csv(xy_data_path)
    stereocalibrator = StereoCalibrator(camera_array, image_points)
    logger.info("Initiating stereocalibration")
    stereo_results = stereocalibrator.stereo_calibrate_all(boards_sampled=10)

    logger.info("stereocalibration complete")
    logger.info("Initializing estimated camera positions based on best daisy-chained stereopairs")
    initializer = CameraArrayInitializer(camera_array, stereo_results)
    camera_array: CameraArray = initializer.get_best_camera_array()
    logger.info("Camera Poses estimated from stereocalibration")

    # should have posed all ports but 4 and 5
    # Using set for posed_cameras to avoid order dependency
    assert set(camera_array.posed_cameras.keys()) == {1, 2, 3, 6}
    assert list(camera_array.unposed_cameras.keys()) == [4, 5]

    # when creating extrinsic parameters shouldn't have camera 5.
    extrinsic_params = camera_array.get_extrinsic_params()
    assert extrinsic_params is not None, "Extrinsic parameters should not be None"
    assert extrinsic_params.shape == (4, 6), "Shape should be (4 posed cameras, 6 params)"

    # camera 5 should not be in the index used for optimization parameter mapping
    assert 5 not in camera_array.posed_port_to_index
    assert 4 not in camera_array.posed_port_to_index

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
    print("end")
    # test_deterministic_consistency()
