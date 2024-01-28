import caliscope.logger

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from caliscope.controller import Controller
from caliscope.gui.vizualize.calibration.capture_volume_visualizer import (
    CaptureVolumeVisualizer,
)

logger = caliscope.logger.get(__name__)


class CaptureVolumeWidget(QWidget):
    def __init__(self, controller: Controller):
        super(CaptureVolumeWidget, self).__init__()

        self.controller = controller

        if not hasattr(self.controller, "capture_volume"):
            self.controller.load_estimated_capture_volume()

        self.visualizer = CaptureVolumeVisualizer(self.controller.capture_volume)
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

        # self.distance_error_summary = QLabel(self.session.quality_controller.distance_error_summary.to_string(index=False))
        self.rmse_summary = QLabel(self.controller.capture_volume.get_rmse_summary())

        # self.recalibrate_btn = QPushButton("Recalibrate")

        self.place_widgets()
        self.connect_widgets()

        self.visualizer.display_points(self.visualizer.min_sync_index)

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.visualizer.scene, stretch=2)
        self.layout().addWidget(self.slider)

        self.grid = QGridLayout()
        self.grid.addWidget(self.rotate_x_plus_btn, 0, 0)
        self.grid.addWidget(self.rotate_x_minus_btn, 1, 0)
        self.grid.addWidget(self.rotate_y_plus_btn, 0, 1)
        self.grid.addWidget(self.rotate_y_minus_btn, 1, 1)
        self.grid.addWidget(self.rotate_z_plus_btn, 0, 2)
        self.grid.addWidget(self.rotate_z_minus_btn, 1, 2)

        self.world_origin_group = QGroupBox()
        self.world_origin_group.setLayout(QVBoxLayout())
        self.world_origin_group.layout().addWidget(self.set_origin_btn)
        self.world_origin_group.layout().addLayout(self.grid)

        self.calibrate_group = QGroupBox()
        self.calibrate_group.setLayout(QVBoxLayout())
        self.calibrate_group.layout().addWidget(self.rmse_summary)
        # self.calibrate_group.layout().addWidget(self.recalibrate_btn)

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.calibrate_group)
        self.hbox.addWidget(self.world_origin_group)
        self.layout().addLayout(self.hbox)

        # self.layout().addWidget(self.recalibrate_btn)

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
        logger.info("Setting origin to board...")
        origin_index = self.slider.value()
        self.controller.set_capture_volume_origin_to_board(origin_index)
        self.visualizer.refresh_scene()

    def rotate_capture_volume(self, direction):
        logger.info(f"Rotating capture volume: {direction}")

        self.controller.rotate_capture_volume(direction)
        self.visualizer.refresh_scene()

    def update_board(self, sync_index):
        logger.info(f"Updating board to sync index {sync_index}")

        self.visualizer.display_points(sync_index)
