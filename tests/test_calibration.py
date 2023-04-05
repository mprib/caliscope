import os
import shutil
from pathlib import Path
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d import __root__
from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates 
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import get_point_estimates
import pytest

TEST_SESSIONS = ["217"]


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
            shutil.copy2(src_item, dst_item)  # Copy file preserving metadata
        elif src_item.is_dir():
            shutil.copytree(src_item, dst_item)



@pytest.fixture(params=TEST_SESSIONS)
def session_path(request, tmp_path):
    """
    Tests will be run based on data stored in tests/sessions, but to avoid overwriting
    or altering test cases,the tested directory will get copied over to a temp
    directory and then that temp directory will be passed on to the calling functions
    """
    original_test_data_path = Path(__root__, "tests", "sessions", request.param)
    tmp_test_data_path = Path(tmp_path,request.param)
    copy_contents(original_test_data_path,tmp_test_data_path)    
    
    return tmp_test_data_path

    
def test_capture_volume_optimization(session_path):
    """
    requires as a baseline a stereocalibrated config.toml file
    """    
    config_path = Path(session_path, "config.toml")
    initializer = CameraArrayInitializer(config_path)
    camera_array = initializer.get_best_camera_array()
    point_data_path = Path(session_path, "point_data.csv")
    point_estimates: PointEstimates = get_point_estimates(camera_array, point_data_path)
    capture_volume = CaptureVolume(camera_array, point_estimates)
    initial_rmse = capture_volume.rmse
    capture_volume.optimize()
    optimized_rmse = capture_volume.rmse

    # rmse should go down after optimization
    for key, rmse in initial_rmse.items():
        assert(rmse>=optimized_rmse[key])


