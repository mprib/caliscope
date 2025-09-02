from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
from caliscope.controller import Controller
from caliscope.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget
import caliscope.logger
import pickle


# capture_volume_pkl = "capture_volume_stage_0_initial.pkl"
# capture_volume_pkl = "capture_volume_stage_1_post_filtering.pkl"
# capture_volume_pkl = "capture_volume_stage_1_post_optimization.pkl"
capture_volume_pkl = "capture_volume_stage_2_post_filtering_then_optimizing.pkl"

logger = caliscope.logger.get(__name__)

app = QApplication(sys.argv)
root = Path(__file__).parent.parent.parent

workspace_dir = Path(root, "tests", "sessions_copy_delete","capture_volume_pre_quality_control")

controller = Controller(workspace_dir)

with open(Path(workspace_dir,capture_volume_pkl), 'rb') as file:
    cap_vol = pickle.load(file)

controller.capture_volume = cap_vol
# controller.load_camera_array()
# controller.load_estimated_capture_volume()
window = CaptureVolumeWidget(controller)
# After filtering - log filtered point counts

logger.info(f"Point counts loaded into Capture Volume Widget:")
logger.info(f"  3D points (obj.shape[0]): {controller.capture_volume.point_estimates.obj.shape[0]}")
logger.info(f"  2D observations (img.shape[0]): {controller.capture_volume.point_estimates.img.shape[0]}")
logger.info(f"  Camera indices length: {len(controller.capture_volume.point_estimates.camera_indices)}")

window.show()

app.exec()
