import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import sys
import math
from pathlib import Path
from threading import Thread, Event
import numpy as np
from queue import Queue
import pandas as pd
from pyxy3d.trackers.tracker_enum import TrackerEnum

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

from pyxy3d.post_processing_pipelines import create_xyz
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

        self.update_recording_folders()

        # select the first element of the QListWidget
        if self.recording_folders.count() > 0:
            self.recording_folders.setCurrentRow(0)
        
        self.tracker_combo= QComboBox()
        
        # Add items to the combo box using the name attribute of the TrackerEnum
        for tracker in TrackerEnum:
            if tracker.name != "CHARUCO":
                self.tracker_combo.addItem(tracker.name, tracker)

        self.vizualizer_title = QLabel(self.viz_title_html)
        self.vis_widget = PlaybackTriangulationWidget(self.camera_array)
        self.process_btn = QPushButton("&Process")
        self.export_btn = QPushButton("&Export")
        
        self.place_widgets()
        self.connect_widgets()
        self.refresh_visualizer(self.recording_folders.currentItem)

    def set_current_xyz(self):
        if self.contains_xyz(self.active_folder):
            self.xyz = pd.read_csv(Path(self.config.session_path, self.active_folder, "xyz.csv"))
        else:
            self.xyz = None
        self.vis_widget.visualizer.set_xyz(self.xyz)
            
        
    def update_recording_folders(self):
        
        if hasattr(self, "recording_folders"):
            self.recording_folders.clear()
        else:
            self.recording_folders = QListWidget()
            
        # create list of recording directories
        dir_list = [p.stem for p in self.config.session_path.iterdir() if p.is_dir()]
        dir_list.remove("calibration")

        # add each folder to the QListWidget
        for folder in dir_list:
            self.recording_folders.addItem(folder)

    def contains_xyz(self, folder_name:str):
        session_path = self.config.session_path
        recording_folders = [self.recording_folders.item(i).text() for i in range(self.recording_folders.count())]
        path = Path(session_path, folder_name, "xyz.csv")
        return path.exists()

    @property
    def active_folder(self):
        if self.recording_folders.currentItem() is not None:
            active_folder:str = self.recording_folders.currentItem().text()
        else:
            active_folder = None
        return active_folder

    @property
    def viz_title_html(self):
        if self.contains_xyz(self.active_folder):
            suffix = "(x,y,z) estimates"
        else:
            suffix = "No processed data"

        title = f"<div align='center'><h2> {self.active_folder}: {suffix} </h2></div>"

        return title

    def place_widgets(self):
        self.setLayout(QHBoxLayout())
        self.left_vbox = QVBoxLayout()
        self.right_vbox = QVBoxLayout()
        self.button_hbox = QHBoxLayout()
        
        self.layout().addLayout(self.left_vbox)
        
        self.left_vbox.addWidget(self.recording_folders)
        self.left_vbox.addWidget(self.tracker_combo)
        self.button_hbox.addWidget(self.process_btn)
        self.button_hbox.addWidget(self.export_btn)
        self.left_vbox.addLayout(self.button_hbox)

        self.layout().addLayout(self.right_vbox, stretch =2)
        self.right_vbox.addWidget(self.vizualizer_title)
        self.right_vbox.addWidget(self.vis_widget, stretch=2)
        
        
    def connect_widgets(self):
        self.recording_folders.currentItemChanged.connect(self.refresh_visualizer)
        self.process_btn.clicked.connect(self.process_current)
                
    def process_current(self):
        recording_path = Path(self.config.session_path, self.active_folder)
        # logger.info(f"{self.tracker_combo.currentData()}")
        tracker_enum = self.tracker_combo.currentData()
        create_xyz(self.config.session_path,recording_path, tracker_enum=tracker_enum)

    def refresh_visualizer(self, item):
        
        # logger.info(f"Item {item.text()} selected and double-clicked.")
        self.vizualizer_title.setText(self.viz_title_html)
        self.update_enabled_disabled()
        self.set_current_xyz()
    
        
    def update_enabled_disabled(self):
        if self.contains_xyz(self.active_folder):
            self.export_btn.setEnabled(True)
            self.process_btn.setEnabled(False)
            self.vis_widget.slider.setEnabled(True)
        else:
            self.export_btn.setEnabled(False)
            self.process_btn.setEnabled(True)
            self.vis_widget.slider.setEnabled(False)
            
            