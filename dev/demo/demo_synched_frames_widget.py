from PySide6.QtWidgets import QApplication
import sys

from pyxy3d.gui.synched_frames_display import SynchedFramesDisplay
from pyxy3d.controller import Controller

from pathlib import Path

from pyxy3d import __root__
from pyxy3d.helper import copy_contents
import pyxy3d.logger
from time import sleep

logger = pyxy3d.logger.get(__name__)
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


