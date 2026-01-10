import logging
from PySide6.QtWidgets import QApplication
import sys

from caliscope.gui.synched_frames_display import SyncedFramesDisplay
from caliscope.controller import Controller
from caliscope.managers.synchronized_stream_manager import SynchronizedStreamManager

from pathlib import Path

from time import sleep
from caliscope.trackers.holistic.holistic_tracker import HolisticTracker

logger = logging.getLogger(__name__)

app = QApplication(sys.argv)

demo_workspace = Path("/home/mprib/caliscope_projects/minimal_project")
# video_dir = demo_workspace / "calibration/extrinsic"
video_dir = demo_workspace / "recordings/sitting"
controller = Controller(demo_workspace)
camera_array = controller.config.get_camera_array()
# charuco = controller.config.get_charuco()
# tracker = CharucoTracker(charuco)
tracker = HolisticTracker()
sync_stream_manager = SynchronizedStreamManager(video_dir, camera_array.cameras, tracker=tracker)

sync_stream_manager.process_streams(fps_target=10)
window = SyncedFramesDisplay(sync_stream_manager)

# need to let synchronizer spin up before able to display frames
while not hasattr(sync_stream_manager.synchronizer, "current_sync_packet"):
    sleep(0.25)

window.show()
window.resize(1050, 1050)
logger.info("About to show window")
sys.exit(app.exec())
