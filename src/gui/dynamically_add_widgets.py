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

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.gui.video_display_gui import MainVideoWindow

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
        
        total_columns = 4
        col = 0
        row = 0
        for i in range(0,2):
            if col > total_columns:
                col = 0
                row = row+1

            vid_window = MainVideoWindow(i) 
            self.gbox.addWidget(vid_window, col, row)
 
            col = col + 1

        self.widget.setLayout(self.gbox)

        #Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)

        self.setGeometry(600, 100, 1000, 900)
        self.setWindowTitle('Scroll Area Demonstration')
        self.show()

        return        
    
    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))


class VideoDisplayWidget(QThread):

    ImageUpdate = pyqtSignal(QImage)
    
    def __init__(self, vid_cap_widget):
        super(VideoDisplayWidget,self).__init__()

        self.vid_cap_widget = vid_cap_widget

    def run(self):
        self.ThreadActive = True

        while self.ThreadActive:
            try:    # takes a moment for capture widget to spin up...don't error out
                self.vid_cap_widget.grab_frame()
                frame = self.vid_cap_widget.frame
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                qt_frame = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format.Format_RGB888)
                Pic = qt_frame.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)

            except AttributeError:
                pass
    def stop(self):
        self.ThreadActive = False
        self.quit()

############### TEST #######################

if __name__ == "__main__":


    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()