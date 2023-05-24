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
from pyxy3d.export import xyz_to_trc

import cv2
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QUrl
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap, QIcon
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
from pyxy3d.session.session import Session
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

        self.sync_index_cursors = {}

        self.update_recording_folders()

        # select the first element of the QListWidget
        if self.recording_folders.count() > 0:
            self.recording_folders.setCurrentRow(0)

        self.tracker_combo = QComboBox()

        # Add items to the combo box using the name attribute of the TrackerEnum
        for tracker in TrackerEnum:
            if tracker.name != "CHARUCO":
                self.tracker_combo.addItem(tracker.name, tracker)

        self.vizualizer_title = QLabel(self.viz_title_html)
        self.vis_widget = PlaybackTriangulationWidget(self.camera_array)
        self.process_current_btn = QPushButton("&Process")
        # self.export_btn = QPushButton("&Export")
        self.open_folder_btn = QPushButton("&Open Folder")

        self.place_widgets()
        self.connect_widgets()
        self.refresh_visualizer()

    def set_current_xyz(self):
        if self.processed_xyz_path.exists():
            self.xyz = pd.read_csv(self.processed_xyz_path)
        else:
            self.xyz = None
        self.vis_widget.set_xyz(self.xyz)

    def update_recording_folders(self):
        if hasattr(self, "recording_folders"):
            self.recording_folders.clear()
        else:
            self.recording_folders = QListWidget()

        # create list of recording directories
        dir_list = [p.stem for p in self.config.session_path.iterdir() if p.is_dir()]
        try:
            dir_list.remove("calibration")
        except:
            pass

        # add each folder to the QListWidget
        for folder in dir_list:
            self.recording_folders.addItem(folder)

    @property
    def processed_subfolder(self):
        subfolder = Path(
            self.config.session_path,
            self.recording_folders.currentItem().text(),
            self.tracker_combo.currentData().name,
        )
        return subfolder

    @property
    def processed_xyz_path(self):
        file_name = f"xyz_{self.tracker_combo.currentData().name}.csv"
        return Path(self.processed_subfolder, file_name)

    def current_selection_processed(self) -> bool:
        """ "
        checks to see if their is a file in the recording directory named `xyz_TRACKERNAME.csv`
        """

        xyz_output = f"xyz_{self.tracker_combo.currentData().name}.csv"
        target_path = Path(self.processed_subfolder, xyz_output)

        return target_path.exists()

    @property
    def active_folder(self):
        if self.recording_folders.currentItem() is not None:
            active_folder: str = self.recording_folders.currentItem().text()
        else:
            active_folder = None
        return active_folder

    @property
    def viz_title_html(self):
        if self.processed_xyz_path.exists():
            suffix = "(x,y,z) estimates"
        else:
            suffix = "(no processed data)"

        title = f"<div align='center'><b>{self.tracker_combo.currentData().name.title()} Tracker: {self.active_folder} {suffix} </b></div>"

        return title

    def place_widgets(self):
        self.setLayout(QHBoxLayout())
        self.left_vbox = QVBoxLayout()
        self.right_vbox = QVBoxLayout()
        self.button_hbox = QHBoxLayout()

        self.layout().addLayout(self.left_vbox)

        self.left_vbox.addWidget(self.recording_folders)
        self.left_vbox.addWidget(self.open_folder_btn)
        self.left_vbox.addWidget(self.tracker_combo)
        self.button_hbox.addWidget(self.process_current_btn)
        # self.button_hbox.addWidget(self.export_btn)
        self.left_vbox.addLayout(self.button_hbox)

        self.layout().addLayout(self.right_vbox, stretch=2)
        self.right_vbox.addWidget(self.vizualizer_title)
        self.right_vbox.addWidget(self.vis_widget, stretch=2)

    def connect_widgets(self):
        self.recording_folders.currentItemChanged.connect(self.refresh_visualizer)
        self.tracker_combo.currentIndexChanged.connect(self.refresh_visualizer)

        self.vis_widget.slider.valueChanged.connect(self.store_sync_index_cursor)

        self.process_current_btn.clicked.connect(self.process_current)
        self.open_folder_btn.clicked.connect(self.open_folder)
        # self.export_btn.clicked.connect(self.export_current_file)

    def store_sync_index_cursor(self, cursor_value):
        if self.processed_xyz_path.exists():
            self.sync_index_cursors[self.processed_xyz_path] = cursor_value
            logger.info(self.sync_index_cursors)
        else:
            # don't bother, doesn't exist
            pass

    def open_folder(self):
        """Opens the currently active folder in a system file browser"""
        if self.active_folder is not None:
            folder_path = Path(self.config.session_path, self.active_folder)
            url = QUrl.fromLocalFile(str(folder_path))
            QDesktopServices.openUrl(url)
        else:
            logger.warn("No folder selected")

    def process_current(self):
        logger.info(f"Beginning to process video files at {self.config.session_path}")
        recording_path = Path(self.config.session_path, self.active_folder)
        # logger.info(f"{self.tracker_combo.currentData()}")
        tracker_enum = self.tracker_combo.currentData()
        logger.info(f"Applying {tracker_enum.name} tracker")

        # a way to receive updates on the progress of the post processing
        self.process_progress = Queue()

        def processing_worker():
            self.disable_all_inputs()

            create_xyz(
                self.config.session_path,
                recording_path,
                tracker_enum=tracker_enum,
                progress_q=self.process_progress,
            )
            trc_path = Path(
                self.processed_xyz_path.parent, self.processed_xyz_path.stem + ".trc"
            )
            logger.info(f"Saving data to {trc_path.parent}")

            # A side effect of the following line is that it also creates a wide labelled csv format
            xyz_to_trc(
                self.processed_xyz_path, self.tracker_combo.currentData().value()
            )
            self.enable_all_inputs()
            self.refresh_visualizer()

        thread = Thread(target=processing_worker, args=(), daemon=True)
        thread.start()

    def refresh_visualizer(self):
        # logger.info(f"Item {item.text()} selected and double-clicked.")
        self.set_current_xyz()
        self.vizualizer_title.setText(self.viz_title_html)
        self.update_enabled_disabled()
        self.update_slider_position()

    def disable_all_inputs(self):
        """used to toggle off all inputs will processing is going on"""
        self.recording_folders.setEnabled(False)
        self.tracker_combo.setEnabled(False)
        # self.export_btn.setEnabled(False)
        self.process_current_btn.setEnabled(False)
        self.vis_widget.slider.setEnabled(False)

    def enable_all_inputs(self):
        """
        after processing completes, swithes everything on again,
        but fine tuning of enable/disable will happen with self.update_enabled_disabled
        """
        self.recording_folders.setEnabled(True)
        self.tracker_combo.setEnabled(True)
        # self.export_btn.setEnabled(True)
        self.process_current_btn.setEnabled(True)
        self.vis_widget.slider.setEnabled(True)

    def update_enabled_disabled(self):
        if self.processed_xyz_path.exists():
            # self.export_btn.setEnabled(True)
            self.process_current_btn.setEnabled(False)
            self.vis_widget.slider.setEnabled(True)

        else:
            # self.export_btn.setEnabled(False)
            self.process_current_btn.setEnabled(True)
            self.vis_widget.slider.setEnabled(False)

    def update_slider_position(self):
        # update slider value to stored value if it exists
        if self.processed_xyz_path in self.sync_index_cursors.keys():
            active_sync_index = self.sync_index_cursors[self.processed_xyz_path]
            self.vis_widget.slider.setValue(active_sync_index)
            self.vis_widget.visualizer.display_points(active_sync_index)
        else:
            pass
