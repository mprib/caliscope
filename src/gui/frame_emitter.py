import logging

logging.basicConfig(filename="frame_emitter.log", filemode="w", level=logging.INFO)

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

    def __init__(self, monocalibrator, pixmap_edge_length=None):
        # pixmap_edge length is from the display window. Keep the display area
        # square to keep life simple.
        super(FrameEmitter, self).__init__()
        self.monocalibrator = monocalibrator
        self.pixmap_edge_length = pixmap_edge_length
        self.rotation_count = monocalibrator.camera.rotation_count

    def run(self):
        self.ThreadActive = True

        while self.ThreadActive:
            # Grab a frame from the queue and broadcast to displays
            self.monocalibrator.grid_frame_ready_q.get()
            frame = self.monocalibrator.grid_frame
            image = self.cv2_to_qlabel(frame)
            pixmap = QPixmap.fromImage(image)

            if self.pixmap_edge_length:
                pixmap = pixmap.scaled(
                    self.pixmap_edge_length,
                    self.pixmap_edge_length,
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            self.ImageBroadcast.emit(pixmap)

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
