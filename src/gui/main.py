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
from src.gui.left_sidebar.sidebar import SideBar

class MainWindow(QMainWindow):
    def __init__(self, session=None):
        super().__init__()
        self.repo = Path(__file__).parent.parent.parent
        if session is not None:
            self.session = session

        app = QApplication.instance()
        screen = app.primaryScreen()
        DISPLAY_WIDTH = screen.size().width()
        DISPLAY_HEIGHT = screen.size().height()

        self.setMinimumSize(DISPLAY_WIDTH * 0.30, DISPLAY_HEIGHT * 0.7)
        self.setWindowTitle("FreeMocap Camera Calibration")
        self.setWindowIcon(QIcon("src/gui/icons/fmc_logo.ico"))
        
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_new_session = QAction("Create &New Session", self)
        file_new_session.triggered.connect(self.open_session)
        file_menu.addAction(file_new_session)

        file_saved_session = QAction("&Open Saved Session", self)
        file_saved_session.triggered.connect(self.open_session)
        file_menu.addAction(file_saved_session)

        view_menu = menu.addMenu("&View")
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.hbox = QHBoxLayout()
        self.central_widget.setLayout(self.hbox)
        
         
    def open_session(self):
        # folder_dialog = QFileDialog
        sessions_directory = str(Path(self.repo, "sessions"))
        session_path = QFileDialog.getExistingDirectory(self,"Select Session Folder", sessions_directory )
        logging.info(f"Opening session located at {session_path}")
        self.session = Session(session_path)
        self.sidebar = SideBar(self.session)
        
        # https://www.youtube.com/watch?v=gGIlLOqRBs4
        # see above for guidance regarding dockable widget, which I think is
        # what I want for the SessionSummary. Also, I think I can rename the 
        # left_side_bar to SessionSummary. And then the true central widget of the
        # QMainWindow can become that actual item of focus, which I think is a good
        # indication that this is the right way to set things up.

        self.sidebar.setMaximumWidth(self.width()/3)
        self.hbox.addWidget(self.sidebar)

    

if __name__ == "__main__":
    repo = Path(__file__).parent.parent.parent
    config_path = Path(repo, "sessions", "high_res_session")
    # print(config_path)
    session = Session(config_path)

    # comment out this next line if you want to save cameras after closing
    # session.delete_all_cam_data() 

    app = QApplication(sys.argv)
    window = MainWindow(session)
    # window = SessionSummary(session)
    window.show()

    app.exec()

