import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import sys
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QSlider,
    QVBoxLayout,
    QWidget,
)
import pandas as pd
from pyxy3d.session import Session
from pyxy3d.gui.vizualize.triangulation.triangulation_visualizer import TriangulationVisualizer
from pyxy3d.cameras.camera_array import CameraArray

class TriangulationWidget(QWidget):
    def __init__(self, camera_array:CameraArray, xyz_history_path:Path):
        super(TriangulationWidget, self).__init__()

        self.camera_array = camera_array
        self.xyz_history = pd.read_csv(xyz_history_path)

        self.visualizer = TriangulationVisualizer(self.camera_array, self.xyz_history)
        # self.visualizer.scene.show()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(self.visualizer.min_sync_index)
        self.slider.setMaximum(self.visualizer.max_sync_index)

        self.setMinimumSize(500, 500)


        self.place_widgets()
        self.connect_widgets()

        # self.visualizer.display_points(self.visualizer.min_sync_index)

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.visualizer.scene)
        self.layout().addWidget(self.slider)


    def connect_widgets(self):
        self.slider.valueChanged.connect(self.visualizer.display_points)



if __name__ == "__main__":

    from PyQt6.QtWidgets import QApplication

    from pyxy3d import __root__
    from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
        get_point_estimates,
    )

    from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
    import pickle

    test_sessions = [
        Path(__root__, "dev", "sample_sessions", "post_triangulation"),
    ]

    test_session_index = 0
    session_path = test_sessions[test_session_index]
    logger.info(f"Loading session {session_path}")
    session = Session(session_path)

    session.load_estimated_capture_volume()

    app = QApplication(sys.argv)
    
    xyz_history_path = Path(session_path,"xyz_history.csv")
    vizr_dialog = TriangulationWidget(session.capture_volume.camera_array,xyz_history_path)
    vizr_dialog.show()

    sys.exit(app.exec())
