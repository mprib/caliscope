import logging
import sys

LOG_FILE = "log\main.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import time
from pathlib import Path
from threading import Thread

from numpy import char
from PyQt6.QtCore import Qt, QDir
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
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
    QToolBar
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.session import Session

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.repo = Path(__file__).parent.parent.parent
        app = QApplication.instance()
        screen = app.primaryScreen()
        DISPLAY_WIDTH = screen.size().width()
        DISPLAY_HEIGHT = screen.size().height()

        self.setMinimumSize(DISPLAY_WIDTH * 0.30, DISPLAY_HEIGHT * 0.7)
        self.setWindowTitle("FreeMocap Camera Calibration")
        self.setWindowIcon(QIcon("src/gui/icons/fmc_logo.ico"))
        
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        new_session = QAction("Create &New Session", self)
        new_session.triggered.connect(self.open_session)
        file_menu.addAction(new_session)

        saved_session = QAction("&Open Saved Session", self)
        saved_session.triggered.connect(self.open_session)
        file_menu.addAction(saved_session)

    def open_session(self):
        # folder_dialog = QFileDialog
        sessions_directory = str(Path(self.repo, "sessions"))
        session_path = QFileDialog.getExistingDirectory(self,"Select Session Folder", sessions_directory )
        self.session = Session(session_path)
        print(session_path)


if __name__ == "__main__":
    repo = Path(__file__).parent.parent.parent
    # config_path = Path(repo, "sessions", "default_res_session")
    config_path = Path(repo, "sessions", "high_res_session")
    # print(config_path)
    # session = Session(config_path)

    # comment out this next line if you want to save cameras after closing
    # session.delete_all_cam_data() 

    app = QApplication(sys.argv)
    window = MainWindow()
    # window = SessionSummary(session)
    window.show()

    app.exec()

