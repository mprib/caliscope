from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer
from pathlib import Path
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d import __root__
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates 
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import get_point_estimates

def test_camera_array_initializer():
    
    session_directory = Path(__root__, "tests", "sessions", "217")
    config_path = Path(session_directory, "config.toml")
    initializer = CameraArrayInitializer(config_path)
    camera_array = initializer.get_best_camera_array()

    assert(isinstance(camera_array, CameraArray))

    
def test_capture_volume_optimization():
    
    session_directory = Path(__root__, "tests", "sessions", "217")
    config_path = Path(session_directory, "config.toml")
    initializer = CameraArrayInitializer(config_path)
    camera_array = initializer.get_best_camera_array()
    point_data_path = Path(session_directory, "point_data.csv")
    point_estimates: PointEstimates = get_point_estimates(camera_array, point_data_path)
    capture_volume = CaptureVolume(camera_array, point_estimates)
    initial_rmse = capture_volume.rmse
    capture_volume.optimize()
    optimized_rmse = capture_volume.rmse

    # rmse should go down after optimization
    for key, rmse in initial_rmse.items():
        assert(rmse>=optimized_rmse[key])


 