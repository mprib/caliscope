from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
from caliscope.controller import Controller
from caliscope.gui.post_processing_widget import PostProcessingWidget

app = QApplication(sys.argv)
workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive - The University of Texas at Austin\research\caliscope\example_project")

controller = Controller(workspace_dir)
controller.load_camera_array()
window = PostProcessingWidget(controller)


window.show()

app.exec()

