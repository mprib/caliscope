
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
from pyxy3d.gui.vizualize.playback_triangulation_widget import PlaybackTriangulationWidget


class PostProcessingWidget(QWidget):
    
    def __init__(self, config:Configurator):
        super(PostProcessingWidget,self).__init__()
        self.config = config
        self.camera_array = self.config.get_camera_array() 

        # create primary elements of interface 
        dir_list = [p.stem for p in self.config.session_path.iterdir() if p.is_dir()]
        dir_list.remove("calibration")
        # add each folder to the QListWidget
        self.recording_folders= QListWidget()

        for folder in dir_list:
            self.recording_folders.addItem(folder)
            
        self.visualizer = PlaybackTriangulationWidget(self.camera_array)
        
        self.place_widgets()
        self.connect_widgets()
        
        
    def place_widgets(self):
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.recording_folders)
        self.layout().addWidget(self.visualizer)
        
    def connect_widgets(self):
        pass