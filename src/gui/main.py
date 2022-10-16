# https://www.youtube.com/watch?v=4BZL3cF_Dww
# This module builds off of the tutorial above to add in widgets to a central
# scroll area. My hope is that this will provide the foundation for the various
# video display widgets while moving through the camera setup calibration
# and capture process.

# There should also, I believe, but a window to assess the quality of the 
# calibration using the Charuco. Is the ultimate reconstruction of the 
# charuco flat and of the same dimensions as the real one? 

import sys
from pathlib import Path

import cv2

from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QMainWindow,
    QLabel, QLineEdit, QCheckBox, QScrollArea, QToolBar,
    QVBoxLayout, QHBoxLayout, QGridLayout, QStatusBar)
from PyQt6.QtMultimedia import QMediaPlayer, QMediaCaptureSession, QVideoFrame
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap, QAction
from numpy import disp


# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera
from src.cameras.real_time_device import RealTimeDevice
from src.gui.camera_config_dialogue import CameraConfigDialog

class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.setMinimumSize(300, 400)

        # set up menu bar
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        window_menu = menu.addMenu("&Window")
        # Close Application
        action_close = QAction("&Close", self)
        action_close.triggered.connect(self.close_app)
        file_menu.addAction(action_close)

        # Add Scroll Area
        action_create_scroll_area = QAction("Create Scroll Area", self)
        action_create_scroll_area.triggered.connect(self.open_scroll_area)
        window_menu.addAction(action_create_scroll_area)

    def close_app(self):
        self.close()

    def open_scroll_area(self):
        self.scroll = QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()                 # Widget that contains the collection of Vertical Box
        self.gbox = QGridLayout()               # The Vertical Box that contains the Horizontal Boxes of  labels and buttons
        
        total_columns = 2
        col = 0
        row = 0

        # this is the call to create the display widgets and is definitely 
        # something that needs to be cleaned up
        # for port in [0,1,3]:
        port = 0
        cam = Camera(port)
        real_time_device = RealTimeDevice(cam)
        display = CameraConfigDialog(real_time_device) 
        # self.gbox.addWidget(vid_window, row, col)

        # col = col + 1
        # if col >= total_columns:
        #     col = 0 
        #     row = row+1
        label = QLabel("Test")
        # display.setFixedSize(300, 300)
        self.gbox.addWidget(display)
        self.widget.setLayout(self.gbox)

        #Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        # self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)

        self.setGeometry(600, 100, 1000, 900)
        self.setWindowTitle('Scroll Area Demonstration')
        self.show()

        return        
    


############### TEST #######################

if __name__ == "__main__":


    app = QApplication(sys.argv)
    # DISPLAY_WIDTH = app.primaryScreen().size().width()
    # DISPLAY_HEIGHT = app.primaryScreen().size().height()
    window = MainWindow()
    window.show()

    app.exec()