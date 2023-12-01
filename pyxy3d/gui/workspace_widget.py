import os
import sys
import subprocess
from pathlib import Path
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSpinBox
from PySide6.QtCore import Slot
from pyxy3d.controller import Controller
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

class WorkspaceSummaryWidget(QWidget):
    def __init__(self, controller:Controller):
        super().__init__()

        self.controller = controller

        self.open_workspace_folder_btn = QPushButton("Open Workspace Directory", self)
        self.load_intrinsics_btn = QPushButton("Load Intrinsic Camera Data")
        self.process_extrinsics_btn = QPushButton("Process Extrinsics") 
        self.calibrate_btn = QPushButton("Calibrate Extrinsics", self)

        self.camera_count_spin = QSpinBox()
        self.camera_count_spin.setValue(self.controller.get_camera_count())
        # Set the layout for the widget
        self.place_widgets()
        self.connect_widgets()
        
        
    def place_widgets(self):
        # Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.camera_count_spin)
        self.layout.addWidget(self.load_intrinsics_btn)
        self.layout.addWidget(self.open_workspace_folder_btn)
        self.layout.addWidget(self.calibrate_btn)
        

    def connect_widgets(self):
        self.open_workspace_folder_btn.clicked.connect(self.open_workspace)  
        self.load_intrinsics_btn.clicked.connect(self.load_intrinsics)
        self.calibrate_btn.clicked.connect(self.on_calibrate_btn_clicked)
        self.camera_count_spin.valueChanged.connect(self.set_camera_count)
        
    def load_intrinsics(self):
        self.controller.load_camera_array()
        self.controller.load_intrinsic_stream_manager()
        
    def on_calibrate_btn_clicked(self):
        logger.info("Calling controller to process extrinsic streams into 2D data")
        self.controller.process_extrinsic_streams(fps_target=100)
        # Call the extrinsic calibration method in the controller
        self.controller.estimate_extrinsics()

    def set_camera_count(self, value):
        self.controller.set_camera_count(value)
        
    def open_workspace(self):
        logger.info(f"Opening workspace within File Explorer...  located at {self.controller.workspace}")
        if sys.platform == 'win32':
            os.startfile(self.controller.workspace)
        elif sys.platform == 'darwin':
            subprocess.run(["open", self.controller.workspace])
        else:  # Linux and Unix-like systems
            subprocess.run(["xdg-open", self.controller.workspace])