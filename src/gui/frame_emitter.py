import logging

logging.basicConfig(filename="synchronizer.log", filemode="w", level=logging.INFO)

import sys
import time
from pathlib import Path

import cv2
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QImage, QPixmap


class FrameEmitter(QThread):
    # establish signals from the frame that will be displayed in real time
    # within the GUI
    ImageBroadcast = pyqtSignal(QPixmap)
    FPSBroadcast = pyqtSignal(int)

    def __init__(self, frame_q, pixmap_edge_length=None):
        # pixmap_edge length is from the display window. Keep the display area
        # square to keep life simple.
        super(FrameEmitter, self).__init__()
        self.frame_q = frame_q
        self.pixmap_edge_length = pixmap_edge_length
        print("Initializing Frame Emitter")

    def run(self):
        self.ThreadActive = True

        while self.ThreadActive:
            # try:  # takes a moment for capture widget to spin up...don't error out

            # Grab a frame from the capture widget and broadcast to displays
            frame = self.frame_q.get()
            image = self.cv2_to_qlabel(frame)
            pixmap = QPixmap.fromImage(image)
            # GUI was crashing I believe due to overloading GUI thread with
            # scaling. Scaling within the emitter resolved the crashes
            if self.pixmap_edge_length:
                pixmap = pixmap.scaled(
                    self.pixmap_edge_length,
                    self.pixmap_edge_length,
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            self.ImageBroadcast.emit(pixmap)
            # grab and broadcast fps
            fps = 0  # TODO: #14 calculate based off of rate of getting from q
            self.FPSBroadcast.emit(fps)

            # throttle rate of broadcast to reduce system overhead
            # if fps == 0:  # Camera likely reconnecting
            #     time.sleep(MIN_SLEEP_TIME)
            # else:
            #     time.sleep(1 / fps)

            # except AttributeError:
            #     pass

    def stop(self):
        self.ThreadActive = False
        self.quit()

    def cv2_to_qlabel(self, frame):
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FlippedImage = cv2.flip(Image, 1)

        qt_frame = QImage(
            FlippedImage.data,
            FlippedImage.shape[1],
            FlippedImage.shape[0],
            QImage.Format.Format_RGB888,
        )
        return qt_frame


if __name__ == "__main__":
    pass

    # not much to look at here... go to camera_config_dialogue.py for test
