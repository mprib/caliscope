import os
import sys
import subprocess
from pathlib import Path
from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout, QPushButton, QSpinBox, QGridLayout, QTextBrowser
from PySide6.QtCore import QFileSystemWatcher, Slot
from pyxy3d.controller import Controller
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

class WorkspaceSummaryWidget(QWidget):
    def __init__(self, controller:Controller):
        super().__init__()

        self.controller = controller
        self.watcher = QFileSystemWatcher()

        self.directory = QLabel(str(self.controller.workspace))
        self.open_workspace_folder_btn = QPushButton("Open", self)

        self.calibrate_btn = QPushButton("Calibrate", self)

        self.camera_count_spin = QSpinBox()
        self.camera_count_spin.setValue(self.controller.get_camera_count())
        self.camera_count_spin.setMaximumWidth(40)

        self.status_HTML = QTextBrowser()
        # Set the layout for the widget
        self.place_widgets()
        self.connect_widgets()

        self.update_status()
        
        
    def place_widgets(self):
        # Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(QLabel("Workspace Directory:"),0,0)
        self.layout.addWidget(self.directory,0,1)
        self.layout.addWidget(self.open_workspace_folder_btn, 0,2)

        self.layout.addWidget(QLabel("Camera Count:"),1,0)
        self.layout.addWidget(self.camera_count_spin,1,1)
        
        self.layout.addWidget(self.status_HTML)

        self.layout.addWidget(self.calibrate_btn,5,0)
        
    def connect_widgets(self):
        self.open_workspace_folder_btn.clicked.connect(self.open_workspace)  
        self.calibrate_btn.clicked.connect(self.on_calibrate_btn_clicked)
        self.camera_count_spin.valueChanged.connect(self.set_camera_count)
        
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

            
    def update_status(self):
        html_content = f"""
        <html>
            <head></head>
            <body>
                <h1>Workspace Status</h1>
                <p>Files present: Yes/No</p>
                <p>Actions possible: List of actions</p>
                <!-- More status information -->
            </body>
        </html> 
        """
        
        self.status_HTML.setHtml(html_content)