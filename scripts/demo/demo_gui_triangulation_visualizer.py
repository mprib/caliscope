from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
from caliscope.controller import Controller
from caliscope.gui.vizualize.playback_triangulation_widget import PlaybackTriangulationWidget
from caliscope.trackers.holistic.holistic_tracker import HolisticTracker
from caliscope import __root__

app = QApplication(sys.argv)
workspace_dir = Path(__root__, "tests", "sessions_copy_delete", "post_optimization")

controller = Controller(workspace_dir)
controller.load_camera_array()
camera_array = controller.camera_array
tracker = HolisticTracker()

xyz_history = Path(workspace_dir, "calibration", "extrinsic", "xyz_CHARUCO.csv")
window = PlaybackTriangulationWidget(
    camera_array=camera_array,
    xyz_history_path=xyz_history,
)

window.show()

app.exec()
