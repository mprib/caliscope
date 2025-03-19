from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
from caliscope.controller import Controller
from caliscope.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget
import caliscope.logger

logger = caliscope.logger.get(__name__)

app = QApplication(sys.argv)
workspace_dir = Path(r"C:\Users\Mac Prible\repos\caliscope\tests\sessions_copy_delete\larger_calibration_post_monocal")
                     

controller = Controller(workspace_dir)
controller.load_camera_array()
controller.load_estimated_capture_volume()
window = CaptureVolumeWidget(controller)
# After filtering - log filtered point counts
logger.info(f"Point counts loaded into Capture Volume Widget:")
logger.info(f"  3D points (obj.shape[0]): {controller.capture_volume.point_estimates.obj.shape[0]}")
logger.info(f"  2D observations (img.shape[0]): {controller.capture_volume.point_estimates.img.shape[0]}")
logger.info(f"  Camera indices length: {len(controller.capture_volume.point_estimates.camera_indices)}")

window.show()

app.exec()
