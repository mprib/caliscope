import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import sys
import math
from pathlib import Path
from threading import Thread, Event
import numpy as np
from queue import Queue

import cv2
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QSizePolicy,
    QWidget,
    QSpinBox,
    QScrollArea,
    QComboBox,
    QCheckBox,
    QTextEdit,
    QLineEdit,
    QDialog,
    QListWidget,
    QGroupBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from pyxy3d.session import Session
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d import __root__
from pyxy3d.recording.video_recorder import VideoRecorder
from pyxy3d.configurator import Configurator
from pyxy3d.gui.vizualize.playback_triangulation_widget import (
    PlaybackTriangulationWidget,
)


class PostProcessingWidget(QWidget):
    def __init__(self, config: Configurator):
        super(PostProcessingWidget, self).__init__()
        self.config = config
        self.camera_array = self.config.get_camera_array()

        # create list of recording directories
        dir_list = [p.stem for p in self.config.session_path.iterdir() if p.is_dir()]
        dir_list.remove("calibration")

        # add each folder to the QListWidget
        self.recording_folders = QListWidget()
        for folder in dir_list:
            self.recording_folders.addItem(folder)

        self.visualizer = PlaybackTriangulationWidget(self.camera_array)
        self.process_btn = QPushButton("Process")
        self.export_btn = QPushButton("Export")
        
        self.place_widgets()
        self.connect_widgets()

    def place_widgets(self):
        self.setLayout(QHBoxLayout())
        self.left_vbox = QVBoxLayout()
        self.right_vbox = QVBoxLayout()
        self.button_hbox = QHBoxLayout()
        
        self.layout().addLayout(self.left_vbox)
        
        self.left_vbox.addWidget(self.recording_folders)
        self.button_hbox.addWidget(self.process_btn)
        self.button_hbox.addWidget(self.export_btn)
        self.left_vbox.addLayout(self.button_hbox)

        self.layout().addLayout(self.right_vbox)
        self.right_vbox.addWidget(self.visualizer)
        
        
    def connect_widgets(self):
        self.recording_folders.itemDoubleClicked.connect(self.set_xyz_history)
        pass

    def set_xyz_history(self, item):
        
        logger.info(f"Item {item.text()} selected and double-clicked.")