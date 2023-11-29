import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
)
from pyxy3d.controller import Controller
import pyxy3d.logger
from pyxy3d import __root__
from pyxy3d.gui.prerecorded_intrinsic_calibration.multiplayback_widget import MultiIntrinsicPlaybackWidget
from pyxy3d.gui.prerecorded_intrinsic_calibration.playback_widget import IntrinsicCalibrationWidget
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.calibration.charuco import Charuco
logger = pyxy3d.logger.get(__name__)

app = QApplication(sys.argv)
# Define the input file path here.
original_workspace_dir = Path(
    __root__, "tests", "sessions", "prerecorded_calibration"
)

# copy_contents(original_workspace_dir, workspace_dir)
workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive\pyxy3d\prerecorded_workflow")

controller = Controller(workspace_dir)
controller.load_camera_array()
controller.load_intrinsic_stream_manager()

window = MultiIntrinsicPlaybackWidget(controller=controller)
# window = IntrinsicCalibrationWidget(controller=controller, port=0)
window.resize(800, 600)
logger.info("About to show window")
window.show()
sys.exit(app.exec())