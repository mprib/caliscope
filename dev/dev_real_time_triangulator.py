from pyxy3d.triangulate.real_time_triangulator import RealTimeTriangulator
from pyxy3d.cameras.camera_array import CameraArray, get_camera_array
from pyxy3d.recording.recorded_stream import RecordedStreamPool
from pyxy3d import get_config
from pyxy3d.calibration.charuco import Charuco, get_charuco

from pathlib import Path

session_path = Path("dev", "sample_sessions","217")

config = get_config(session_path)

config_path = Path(session_path,"config.toml")

charuco:Charuco = get_charuco(config_path)
camera_array:CameraArray=get_camera_array(config)

real_time_triangulator = RealTimeTriangulator(camera_array)