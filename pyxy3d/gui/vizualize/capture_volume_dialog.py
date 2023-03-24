import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

import sys
from pathlib import Path
from threading import Thread

import cv2
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyxy3d.gui.vizualize.capture_volume_visualizer import CaptureVolumeVisualizer

class CaptureVolumeDialog(QWidget):
    def __init__(self, CaptureVolumeVisualizer):
        super(CaptureVolumeDialog, self).__init__()
        self.visualizer = CaptureVolumeVisualizer
        # self.visualizer.scene.show()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(self.visualizer.min_sync_index)
        self.slider.setMaximum(self.visualizer.max_sync_index)
        self.set_origin_btn = QPushButton("Set Origin")

        self.setMinimumSize(500,500)
        
        self.place_widgets()
        self.connect_widgets()

        self.visualizer.display_points(self.visualizer.min_sync_index)
        
        
    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.visualizer.scene)
        self.layout().addWidget(self.slider)
        self.layout().addWidget(self.set_origin_btn)
        # self.visualizer.begin()

    def connect_widgets(self):
        self.slider.valueChanged.connect(self.visualizer.display_points)
        self.set_origin_btn.clicked.connect(self.log_board_points)

    def log_board_points(self):
        logger.info(f"{self.visualizer.single_board_points}")
    def update_board(self, sync_index):
        
        logger.info(f"Updating board to sync index {sync_index}")
        
        self.visualizer.display_points(sync_index)
if __name__ == "__main__":

    from PyQt6.QtWidgets import QApplication

    from pyxy3d import __root__
    from pyxy3d.cameras.camera_array_builder_deprecate import CameraArrayBuilder
    from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import get_point_estimates

    from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
    import pickle
    
    # session_directory = Path(__root__,  "tests", "2_cameras_linear")
    # session_directory = Path(__root__,  "tests", "tripod")
    # session_directory = Path(__root__,  "tests", "2_cameras_90_deg")
    # session_directory = Path(__root__,  "tests", "2_cameras_180_deg")
    # session_directory = Path(__root__,  "tests", "3_cameras_triangular")
    # session_directory = Path(__root__,  "tests", "3_cameras_middle")
    # session_directory = Path(__root__,  "tests", "4_cameras_beginning")
    session_directory = Path(__root__,  "tests", "4_cameras_endofday")
    # session_directory = Path(__root__,  "tests", "4_cameras_nonoverlap")
    # session_directory = Path(__root__,  "tests", "3_cameras_linear")
    # session_directory = Path(__root__,  "tests", "3_cameras_midlinear")
    # session_directory = Path(__root__,  "tests", "just_checking")


    # saved_CV_path = Path(session_directory, "capture_volume_stage_1_optimized.pkl") 
    saved_CV_path = Path(session_directory, "capture_volume_stage_1_new_origin.pkl") 
    with open(saved_CV_path, "rb") as f:
        capture_volume:CaptureVolume = pickle.load(f)

    app = QApplication(sys.argv)
    vizr = CaptureVolumeVisualizer(capture_volume = capture_volume)
    # vizr = CaptureVolumeVisualizer(camera_array = capture_volume.camera_array)


    vizr_dialog = CaptureVolumeDialog(vizr)
    vizr_dialog.show()

    sys.exit(app.exec())