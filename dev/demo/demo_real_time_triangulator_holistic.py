
""""
THINGS ARE WORKING ON APR 16 AT 6:57 AM.
In the future I think I want to change it so that the charuco tracker factory is not
a default, but something that has to be made explicit....
"""
# %%
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)
import sys
from pathlib import Path
from queue import Queue
from time import sleep

from PyQt6.QtWidgets import QApplication

from pyxy3d import __root__
from pyxy3d.cameras.camera_array import (CameraArray, CameraData,
                                         get_camera_array)
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.configurator import Configurator
from pyxy3d.gui.vizualize.realtime_triangulation_widget import \
    RealTimeTriangulationWidget
from pyxy3d.interface import FramePacket, PointPacket, SyncPacket
from pyxy3d.session import Session
from pyxy3d.trackers.charuco_tracker import Charuco, CharucoTracker,CharucoTrackerFactory
from pyxy3d.trackers.holistic_tracker import HolisticTracker, HolisticTrackerFactory
from pyxy3d.triangulate.sync_packet_triangulator import SyncPacketTriangulator

app = QApplication(sys.argv)
session_path = Path(__root__,"dev", "sample_sessions", "real_time")

session = Session(session_path)
# session.load_cameras()
# session.charuco
# tracker_factory = CharucoTrackerFactory()
tracker_factory = HolisticTrackerFactory()

session.load_streams(tracker_factory=tracker_factory) 
session.adjust_resolutions()

config = Configurator(session_path)
camera_array = config.get_camera_array()

logger.info(f"Creating RecordedStreamPool")
# stream_pool = RecordedStreamPool(session_path, tracker_factory=charuco_tracker_factory, fps_target=100)
logger.info("Creating Synchronizer")
syncr = Synchronizer(session.streams, fps_target=12)

real_time_triangulator = SyncPacketTriangulator(camera_array, syncr)
xyz_queue = Queue()

real_time_triangulator.subscribe(xyz_queue)

real_time_widget = RealTimeTriangulationWidget(camera_array,xyz_queue)
 
real_time_widget.show()
sys.exit(app.exec())