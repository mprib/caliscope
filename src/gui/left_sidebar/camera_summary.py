import logging
import sys

LOG_FILE = "log\camera_summary.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import time
from pathlib import Path
from threading import Thread

from numpy import char
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.session import Session
from src.gui.left_sidebar.camera_table import CameraTable

class CameraSummary(QWidget):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.cams_in_process = False
        self.hbox = QHBoxLayout()
        self.setLayout(self.hbox)

        left_vbox = QVBoxLayout()

        self.camera_table = CameraTable(self.session)
        self.camera_table.setFixedSize(250, 150)
        left_vbox.addWidget(self.camera_table)
        self.connect_cams_btn = QPushButton("&Connect to Cameras")

        if self.session.camera_count() == 0:
            self.connect_cams_btn.setEnabled(False)
            
        left_vbox.addWidget(self.connect_cams_btn)
        self.find_cams_btn = QPushButton("&Find Additional Cameras")
        left_vbox.addWidget(self.find_cams_btn)

        self.open_cameras_btn = QPushButton("Open Cameras")  # this button is invisible

        self.hbox.addLayout(left_vbox)
        
        # self.open_cameras_btn.clicked.connect(self.open_cams)
        self.connect_cams_btn.clicked.connect(self.connect_cams)
        self.find_cams_btn.clicked.connect(self.find_additional_cams)

    def connect_cams(self):
        def connect_cam_worker():
            self.cams_in_process = True
            logging.info("Loading Cameras")
            self.session.load_cameras()
            logging.info("Loading streams")
            self.session.load_stream_tools()
            logging.info("Adjusting resolutions")
            self.session.adjust_resolutions()
            logging.info("Loading monocalibrators")
            self.session.load_monocalibrators()
            logging.info("Updating Camera Table")
            self.camera_table.update_data()

            # trying to call open_cams() directly created a weird bug that
            # may be due to all the different threads. This seemed to kick it
            # back to the main thread...
            # self.summary.open_cameras_btn.click()

            self.cams_in_process = False

        if not self.cams_in_process:
            print("Connecting to cameras...This may take a moment.")
            self.connect = Thread(target=connect_cam_worker, args=(), daemon=True)
            self.connect.start()
        else:
            print("Cameras already connected or in process.")

    def find_additional_cams(self):
        def find_cam_worker():

            self.session.find_additional_cameras()
            logging.info("Loading streams")
            self.session.load_stream_tools()
            logging.info("Loading monocalibrators")
            self.session.load_monocalibrators()
            logging.info("Adjusting resolutions")
            self.session.adjust_resolutions()
            logging.info("Updating Camera Table")
            self.camera_table.update_data()

            self.open_cameras_btn.click()

            self.cams_in_process = False
            self.connect_cams_btn.setEnabled(True)
            
        if not self.cams_in_process:
            print("Searching for additional cameras...This may take a moment.")
            self.find = Thread(target=find_cam_worker, args=(), daemon=True)
            self.find.start()
        else:
            print("Cameras already connected or in process.")
        
if __name__ == "__main__":
    repo = Path(__file__).parent.parent.parent.parent
    config_path = Path(repo, "sessions", "high_res_session")
    
    session = Session(config_path)
    print(session.config)
    app = QApplication(sys.argv)
    camera_summary = CameraSummary(session)
    camera_summary.show()
    sys.exit(app.exec())