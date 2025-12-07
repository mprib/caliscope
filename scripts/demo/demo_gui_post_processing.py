from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
from caliscope.controller import Controller
from caliscope.gui.post_processing_widget import PostProcessingWidget

app = QApplication(sys.argv)

workspace_dir = Path("/home/mprib/caliscope_projects/markerless_calibration_data/caliscope_version")

controller = Controller(workspace_dir)
controller.load_camera_array()
window = PostProcessingWidget(controller)


window.show()

app.exec()
