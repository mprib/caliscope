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

from calicam.session import Session
from calicam.gui.left_sidebar.camera_table import CameraTable

class CameraSummary(QWidget):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.cams_in_process = False
        self.hbox = QHBoxLayout()
        self.setLayout(self.hbox)

        vbox = QVBoxLayout()

        self.cam_count_box = QHBoxLayout()
        vbox.addLayout(self.cam_count_box)
        self.cam_count_box.addWidget(QLabel("Connected Cameras Count:"))
        self.connected_cam_count = QLabel("0")
        self.cam_count_box.addWidget(self.connected_cam_count)

        self.camera_table = CameraTable(self.session)
        self.camera_table.setFixedSize(250, 150)
        vbox.addWidget(self.camera_table)

        self.hbox.addLayout(vbox)
        

if __name__ == "__main__":
    repo = Path(__file__).parent.parent.parent.parent
    config_path = Path(repo, "sessions", "high_res_session")
    
    session = Session(config_path)
    print(session.config)
    app = QApplication(sys.argv)
    camera_summary = CameraSummary(session)
    camera_summary.show()
    sys.exit(app.exec())