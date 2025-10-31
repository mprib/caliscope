import logging
from pathlib import Path

import numpy as np
import pytest

from caliscope import __root__
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.configurator import Configurator
from caliscope.helper import copy_contents

logger = logging.getLogger(__name__)


# Define the test to be parameterized over rotation directions
@pytest.mark.parametrize("direction", ["x+", "x-", "y+", "y-", "z+", "z-"])
def test_rotation_invariance(direction: str):
    """
    Tests that applying four 90-degree rotations around an axis returns
    the capture volume to its original state. Also verifies that intermediate
    states (90, 180, 270 degrees) are different from the original state.
    """
    # 1. SETUP
    # Use the clean, post-optimization data as a starting point.
    source_session_path = Path(__root__, "tests", "sessions", "post_optimization")
    test_session_path = Path(__root__, "tests", "sessions_copy_delete", "rotation_test")
    copy_contents(source_session_path, test_session_path)

    config = Configurator(test_session_path)
    camera_array = config.get_camera_array()
    point_estimates = config.load_point_estimates_from_toml()

    capture_volume = CaptureVolume(camera_array, point_estimates)

    # 2. STORE INITIAL STATE
    initial_points = capture_volume.get_xyz_points().copy()
    initial_transforms = {
        port: cam.transformation.copy() for port, cam in capture_volume.camera_array.posed_cameras.items()
    }

    # 3. EXECUTE & ASSERT
    for i in range(1, 5):
        logger.info(f"Applying rotation {i} in direction '{direction}'...")

        # This method doesn't exist yet. The test will fail here.
        capture_volume.rotate(direction)

        current_points = capture_volume.get_xyz_points()
        current_transforms = {
            port: cam.transformation for port, cam in capture_volume.camera_array.posed_cameras.items()
        }

        # After 1, 2, or 3 rotations (90, 180, 270 deg)
        if i < 4:
            # Assert that the points have changed
            assert not np.allclose(
                initial_points, current_points
            ), f"Points should not be the same after {i * 90} degrees"

            # Assert that camera transforms have changed
            for port in initial_transforms:
                assert not np.allclose(
                    initial_transforms[port], current_transforms[port]
                ), f"Transform for camera {port} should not be the same after {i * 90} degrees"

        # After 4 rotations (360 deg)
        else:
            # Assert that the points have returned to the original state
            assert np.allclose(
                initial_points, current_points, atol=1e-6
            ), "Points should return to original state after 360 degrees"

            # Assert that camera transforms have returned to the original state
            for port in initial_transforms:
                assert np.allclose(
                    initial_transforms[port], current_transforms[port], atol=1e-6
                ), f"Transform for camera {port} should return to original after 360 degrees"

    logger.info(f"Rotation test passed for direction '{direction}'.")


if __name__ == "__main__":
    test_rotation_invariance("x-")
