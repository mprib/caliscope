from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
from caliscope.controller import Controller
from caliscope.gui.post_processing_widget import PostProcessingWidget
from caliscope.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget

app = QApplication(sys.argv)
workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive\caliscope\4_cam_prerecorded_practice_working")
controller = Controller(workspace_dir)
# controller.load_camera_array()
controller.load_estimated_capture_volume()
window = CaptureVolumeWidget(controller)


window.show()

app.exec()

