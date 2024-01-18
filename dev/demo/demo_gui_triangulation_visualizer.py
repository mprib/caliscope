from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
from pyxy3d.controller import Controller
from pyxy3d.gui.vizualize.playback_triangulation_widget import PlaybackTriangulationWidget

app = QApplication(sys.argv)
workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive - The University of Texas at Austin\research\pyxy3d\example_project")

controller = Controller(workspace_dir)
controller.load_camera_array()
camera_array = controller.camera_array

xyz_history = Path(workspace_dir, "recordings","test_recording", "HOLISTIC", "xyz_HOLISTIC.csv")
window = PlaybackTriangulationWidget(camera_array=camera_array,xyz_history_path=xyz_history)

window.show()

app.exec()

