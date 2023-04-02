import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import sys
import numpy as np
from pathlib import Path
from threading import Thread

import cv2
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QGridLayout,
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
from pyxy3d.gui.widgets import NavigationBarBackFinish


class CaptureVolumeWidget(QWidget):
    def __init__(self, session: Session):
        super(CaptureVolumeWidget, self).__init__()

        self.session = session
        self.visualizer = CaptureVolumeVisualizer(self.session.capture_volume)
        # self.visualizer.scene.show()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(self.visualizer.min_sync_index)
        self.slider.setMaximum(self.visualizer.max_sync_index)
        self.set_origin_btn = QPushButton("Set Origin")

        self.setMinimumSize(500, 500)

        self.rotate_x_plus_btn = QPushButton("X+")
        self.rotate_x_minus_btn = QPushButton("X-")
        self.rotate_y_plus_btn = QPushButton("Y+")
        self.rotate_y_minus_btn = QPushButton("Y-")
        self.rotate_z_plus_btn = QPushButton("Z+")
        self.rotate_z_minus_btn = QPushButton("Z-")

        self.distance_error_summary = QLabel(self.session.quality_controller.distance_error_summary.to_string(index=False))
        self.rmse_summary = QLabel(self.session.capture_volume.get_rmse_summary())
        

        self.navigation_bar = NavigationBarBackFinish()

        self.place_widgets()
        self.connect_widgets()

        self.visualizer.display_points(self.visualizer.min_sync_index)

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.visualizer.scene)
        self.layout().addWidget(self.slider)
        self.layout().addWidget(self.set_origin_btn)
        # self.visualizer.begin()

        self.grid = QGridLayout()
        self.layout().addLayout(self.grid)
        self.grid.addWidget(self.rotate_x_plus_btn, 0, 0)
        self.grid.addWidget(self.rotate_x_minus_btn, 1, 0)
        self.grid.addWidget(self.rotate_y_plus_btn, 0, 1)
        self.grid.addWidget(self.rotate_y_minus_btn, 1, 1)
        self.grid.addWidget(self.rotate_z_plus_btn, 0, 2)
        self.grid.addWidget(self.rotate_z_minus_btn, 1, 2)

        # thie distance error summary here is ugly and needs to be in a table,
        # But thats out of scope for my current objectives.
        self.hbox_summary = QHBoxLayout()
        self.hbox_summary.addWidget(self.rmse_summary)  
        self.hbox_summary.addWidget(self.distance_error_summary)  
        self.layout().addLayout(self.hbox_summary)

        self.layout().addWidget(self.navigation_bar)


    def connect_widgets(self):
        self.slider.valueChanged.connect(self.visualizer.display_points)
        self.set_origin_btn.clicked.connect(self.set_origin_to_board)
        self.rotate_x_plus_btn.clicked.connect(lambda: self.rotate_capture_volume("x+"))
        self.rotate_x_minus_btn.clicked.connect(
            lambda: self.rotate_capture_volume("x-")
        )
        self.rotate_y_plus_btn.clicked.connect(lambda: self.rotate_capture_volume("y+"))
        self.rotate_y_minus_btn.clicked.connect(
            lambda: self.rotate_capture_volume("y-")
        )
        self.rotate_z_plus_btn.clicked.connect(lambda: self.rotate_capture_volume("z+"))
        self.rotate_z_minus_btn.clicked.connect(
            lambda: self.rotate_capture_volume("z-")
        )

    def set_origin_to_board(self):
        self.session.capture_volume.set_origin_to_board(
            self.slider.value(), self.session.charuco
        )
        self.visualizer.refresh_scene()
        self.session.save_capture_volume()

    def rotate_capture_volume(self, direction):
        transformations = {
            "x+": np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
            ),
            "x-": np.array(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float
            ),
            "y+": np.array(
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float
            ),
            "y-": np.array(
                [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=float
            ),
            "z+": np.array(
                [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
            ),
            "z-": np.array(
                [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
            ),
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
    from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
        get_point_estimates,
    )

    from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
    import pickle

    test_sessions = [
        Path(__root__, "tests", "217"),
        Path(__root__, "tests", "2_cameras_linear"),
        Path(__root__, "tests", "2_cameras_90_deg"),
        Path(__root__, "tests", "2_cameras_180_deg"),
        Path(__root__, "tests", "3_cameras_triangular"),
        Path(__root__, "tests", "3_cameras_middle"),
        Path(__root__, "tests", "4_cameras_beginning"),
        Path(__root__, "tests", "4_cameras_endofday"),
        Path(__root__, "tests", "4_cameras_nonoverlap"),
        Path(__root__, "tests", "4_cameras_nonoverlap"),
        Path(__root__, "tests", "3_cameras_midlinear"),
    ]

    test_session_index = 1
    session_path = test_sessions[test_session_index]
    logger.info(f"Loading session {session_path}")
    session = Session(session_path)

    # session.build_capture_volume_from_stereopairs()
    
    session.estimate_extrinsics()
    # session.load_estimated_capture_volume()

    app = QApplication(sys.argv)

    vizr_dialog = CaptureVolumeWidget(session)
    vizr_dialog.show()

    sys.exit(app.exec())
