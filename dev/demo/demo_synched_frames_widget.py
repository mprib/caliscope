from PySide6.QtWidgets import QApplication
import sys

from caliscope.gui.synched_frames_display import SynchedFramesDisplay
from caliscope.controller import Controller

from pathlib import Path

from caliscope import __root__
from caliscope.helper import copy_contents
import caliscope.logger
from time import sleep

logger = caliscope.logger.get(__name__)
app = QApplication(sys.argv)
# Define the input file path here.
original_workspace_dir = Path(__root__, "tests", "sessions", "mediapipe_calibration")
workspace_dir = Path(__root__, "tests", "sessions_copy_delete", "4_cam_record")

copy_contents(original_workspace_dir, workspace_dir)

controller = Controller(workspace_dir)
controller.load_camera_array()
controller.load_extrinsic_stream_manager()

controller.extrinsic_stream_manager.process_streams(fps_target=100)
window = SynchedFramesDisplay(controller.extrinsic_stream_manager)
# need to let synchronizer spin up before able to display frames
# need to let synchronizer spin up before able to display frames

while not hasattr(controller.extrinsic_stream_manager.synchronizer, "current_sync_packet"):
    sleep(0.25)

window.show()
window.resize(800, 600)
logger.info("About to show window")
sys.exit(app.exec())


