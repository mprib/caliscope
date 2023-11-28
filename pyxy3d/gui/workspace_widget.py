import os
import sys
import subprocess
from pathlib import Path
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Slot
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

class WorkspaceSummaryWidget(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        self.open_workspace_folder_btn = QPushButton("Open Workspace Directory", self)
        self.process_extrinsics_btn = QPushButton("Process Extrinsics", self)
        self.calibrate_btn = QPushButton("Calibrate Extrinsics", self)


        # Set the layout for the widget
        self.place_widgets()
        self.connect_widgets()
        
        
    def place_widgets(self):
        # Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.open_workspace_folder_btn)
        self.layout.addWidget(self.process_extrinsics_btn)
        self.layout.addWidget(self.calibrate_btn)
        

    
    def connect_widgets(self):
        self.open_workspace_folder_btn.clicked.connect(self.open_workspace)  
        self.calibrate_btn.clicked.connect(self.on_calibrate_btn_clicked)
        self.process_extrinsics_btn.clicked.connect(self.on_process_extrinsics_clicked)
        
        
    @Slot()
    def on_calibrate_btn_clicked(self):
        # Call the extrinsic calibration method in the controller
        self.controller.estimate_extrinsics()

    @Slot()
    def on_process_extrinsics_clicked(self):
        logger.info("Calling controller to process extrinsic streams into 2D data")
        self.controller.process_extrinsic_streams(fps_target=100)


    def open_workspace(self):
        logger.info(f"Opening workspace within File Explorer...  located at {self.controller.workspace}")
        if sys.platform == 'win32':
            os.startfile(self.controller.workspace)
        elif sys.platform == 'darwin':
            subprocess.run(["open", self.controller.workspace])
        else:  # Linux and Unix-like systems
            subprocess.run(["xdg-open", self.controller.workspace])