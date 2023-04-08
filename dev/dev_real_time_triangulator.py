import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.triangulate.real_time_triangulator import RealTimeTriangulator
from pyxy3d.cameras.camera_array import CameraArray, get_camera_array
from pyxy3d.recording.recorded_stream import RecordedStreamPool
from pyxy3d.calibration.charuco import Charuco, get_charuco
from pyxy3d.configurator import Configurator

from pathlib import Path

session_path = Path("tests", "sessions","post_optimization")

config = Configurator(session_path)


charuco:Charuco = config.get_charuco()
camera_array:CameraArray=config.get_camera_array()

logger.info(f"Creating RecordedStreamPool")
stream_pool = RecordedStreamPool(session_path, charuco=charuco)
logger.info("Creating Synchronizer")
syncr = Synchronizer(stream_pool.streams, fps_target=None)

real_time_triangulator = RealTimeTriangulator(camera_array, syncr)