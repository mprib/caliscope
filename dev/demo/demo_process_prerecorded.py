
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)
from time import sleep

from pyxy3d.configurator import Configurator
from pathlib import Path
from pyxy3d import __root__
from pyxy3d.trackers.holistic_tracker import HolisticTrackerFactory, HolisticTracker
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.recording.recorded_stream import RecordedStreamPool
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.triangulate.real_time_triangulator import RealTimeTriangulator

session_path = Path(__root__, "dev", "sample_sessions", "recordings_to_process")
recording_path = Path(session_path, "recording_1")
config = Configurator(session_path)
# origin_sync_index = config.dict["capture_volume"]["origin_sync_index"]

tracker_factory = HolisticTrackerFactory()

camera_array: CameraArray = config.get_camera_array()

logger.info(f"Creating RecordedStreamPool")
stream_pool = RecordedStreamPool(recording_path, tracker_factory=tracker_factory, fps_target=12)
logger.info("Creating Synchronizer")
syncr = Synchronizer(stream_pool.streams, fps_target=12)


#### Basic code for interfacing with in-progress RealTimeTriangulator
#### Just run off of saved point_data.csv for development/testing
real_time_triangulator = RealTimeTriangulator(camera_array, syncr, output_directory=recording_path)
stream_pool.play_videos()
while real_time_triangulator.running:
    sleep(1)