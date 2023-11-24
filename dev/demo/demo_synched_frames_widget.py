from PySide6.QtWidgets import QApplication
import sys

from pyxy3d.gui.synched_frames_widget import SynchedFramesWidget
from pyxy3d.controller import Controller

from pathlib import Path

from pyxy3d import __root__
from pyxy3d.helper import copy_contents
import pyxy3d.logger


logger = pyxy3d.logger.get(__name__)
app = QApplication(sys.argv)
# Define the input file path here.
original_workspace_dir = Path(__root__, "tests", "sessions", "4_cam_recording")
workspace_dir = Path(__root__, "tests", "sessions_copy_delete", "4_cam_recording")

copy_contents(original_workspace_dir, workspace_dir)

controller = Controller(workspace_dir)
controller.process_extrinsic_streams()

window = SynchedFramesWidget(controller)
window.resize(800, 600)
logger.info("About to show window")
window.show()
sys.exit(app.exec())


