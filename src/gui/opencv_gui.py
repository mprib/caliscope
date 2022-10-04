# At this point I can't even recall where I copied this code from, but it is what
# actually works and drives home to me that I need to develop a better understanding
# of threads, signals, slots, events, all that as it relates to GUI development

# Moving on to pythonguis.com tutorials for just that. But this code works...

import sys

from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, 
    QLabel, QLineEdit, 
    QVBoxLayout, QHBoxLayout, QGridLayout
)
from PyQt6.QtMultimedia import QMediaPlayer, QMediaCaptureSession, QVideoFrame
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap

# from PyQt6.QtGui import * 
# from PyQt6.QtWidgets import *
# from PyQt6.QtCore import *

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
for p in sys.path:
    print(p)

from src.concurrency_tutorial.video_stream_widget import VideoCaptureWidget
import cv2

class MainWindow(QWidget):
    def __init__(self, vid_cap_widget):
        super(MainWindow, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.vid_display = VideoDisplayWidget(vid_cap_widget)

        self.vid_display.start()
        self.vid_display.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.vid_display.stop()

class VideoDisplayWidget(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def __init__(self, vid_cap_widget):
        super(VideoDisplayWidget,self).__init__()

        self.vid_cap_widget = vid_cap_widget

    def run(self):
        self.ThreadActive = True
        # Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            try:    # takes a moment for capture widget to spin up...don't error out
                self.vid_cap_widget.grab_frame()
                frame = self.vid_cap_widget.raw_frame
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

if __name__ == "__main__":
    # create a camera widget to pull in a thread of frames
    # these are currently processed by mediapipe but don't have to be
    test_cam_widget = VideoCaptureWidget(0,1080,640)

    App = QApplication(sys.argv)
    Root = MainWindow(test_cam_widget)
    Root.show()
    sys.exit(App.exec())