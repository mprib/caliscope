from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
from pyxy3d.controller import Controller
from pyxy3d.gui.post_processing_widget import PostProcessingWidget

app = QApplication(sys.argv)
workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive\pyxy3d\4_cam_prerecorded_practice_working")
controller = Controller(workspace_dir)
controller.load_camera_array()
window = PostProcessingWidget(controller)


window.show()

app.exec()

