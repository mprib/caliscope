from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
from caliscope.controller import Controller
from caliscope.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget
import logging
import caliscope.logger
from caliscope import __root__


caliscope.logger.setup_logging()
logger = logging.getLogger(__name__)

app = QApplication(sys.argv)
root = Path(__file__).parent.parent.parent

workspace_dir = Path(root, "tests", "sessions_copy_delete", "capture_volume_pre_quality_control")

workspace_dir = Path(__root__, "tests", "sessions_copy_delete", "post_optimization")
controller = Controller(workspace_dir)

controller.load_camera_array()
controller.load_estimated_capture_volume()

window = CaptureVolumeWidget(controller)
# After filtering - log filtered point counts

logger.info("Point counts loaded into Capture Volume Widget:")
logger.info(f"  3D points (obj.shape[0]): {controller.capture_volume.point_estimates.obj.shape[0]}")
logger.info(f"  2D observations (img.shape[0]): {controller.capture_volume.point_estimates.img.shape[0]}")
logger.info(f"  Camera indices length: {len(controller.capture_volume.point_estimates.camera_indices)}")

window.show()

app.exec()
