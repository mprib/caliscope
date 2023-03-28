import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

import sys
import numpy as np
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

from pyxy3d.session import Session
from pyxy3d.gui.vizualize.capture_volume_visualizer import CaptureVolumeVisualizer

class CaptureVolumeDialog(QWidget):
    def __init__(self, session:Session):
        super(CaptureVolumeDialog, self).__init__()
        self.session = session
        self.visualizer = CaptureVolumeVisualizer(self.session.capture_volume)
        # self.visualizer.scene.show()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(self.visualizer.min_sync_index)
        self.slider.setMaximum(self.visualizer.max_sync_index)
        self.set_origin_btn = QPushButton("Set Origin")

        self.setMinimumSize(500,500)
       
        self.rotate_x_plus_btn = QPushButton("Rotate X+") 
        self.rotate_x_minus_btn = QPushButton("Rotate X-") 

        self.place_widgets()
        self.connect_widgets()

        self.visualizer.display_points(self.visualizer.min_sync_index)
        
        
    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.visualizer.scene)
        self.layout().addWidget(self.slider)
        self.layout().addWidget(self.set_origin_btn)
        # self.visualizer.begin()
        self.layout().addWidget(self.rotate_x_plus_btn)
        self.layout().addWidget(self.rotate_x_minus_btn)

    def connect_widgets(self):
        self.slider.valueChanged.connect(self.visualizer.display_points)
        self.set_origin_btn.clicked.connect(self.set_origin_to_board)
        self.rotate_x_plus_btn.clicked.connect(lambda: self.rotate_capture_volume("x+"))
        self.rotate_x_minus_btn.clicked.connect(lambda: self.rotate_capture_volume("x-"))

    def set_origin_to_board(self):
        self.session.capture_volume.set_origin_to_board(self.slider.value(), self.session.charuco)       
        self.visualizer.refresh_scene()
        self.session.save_capture_volume()

    def rotate_capture_volume(self, direction):
        transformations ={"x+": np.array([[1,0,0,0],
                                          [0,0,1,0],
                                          [0,-1,0,0],
                                          [0,0,0,1]],dtype=float),
                          "x-": np.array([[1,0,0,0],
                                          [0,0,-1,0],
                                          [0,1,0,0],
                                          [0,0,0,1]],dtype=float),

                            
        }
        self.session.capture_volume.shift_origin(transformations[direction])
        self.visualizer.refresh_scene()
        self.session.save_capture_volume()
        
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
    session_directory = Path(__root__,  "tests", "4_cameras_beginning")
    # session_directory = Path(__root__,  "tests", "4_cameras_endofday")
    # session_directory = Path(__root__,  "tests", "4_cameras_nonoverlap")
    # session_directory = Path(__root__,  "tests", "4_cameras_nonoverlap")
    # session_directory = Path(__root__,  "tests", "3_cameras_linear")
    # session_directory = Path(__root__,  "tests", "3_cameras_midlinear")
    # session_directory = Path(__root__,  "tests", "just_checking")


    # saved_CV_path = Path(session_directory, "capture_volume_stage_1_optimized.pkl") 
    # saved_CV_path = Path(session_directory, "capture_volume_stage_1.pkl") 
    # with open(saved_CV_path, "rb") as f:
        # capture_volume:CaptureVolume = pickle.load(f)

        
    session = Session(session_directory)
    session.load_configured_capture_volume()
    

    app = QApplication(sys.argv)
    # vizr = CaptureVolumeVisualizer(session)
    # vizr = CaptureVolumeVisualizer(camera_array = capture_volume.camera_array)


    vizr_dialog = CaptureVolumeDialog(session)
    vizr_dialog.show()

    sys.exit(app.exec())