import logging

LOG_FILE = "log/frame_emitter.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from datetime import datetime
from pathlib import Path

import cv2
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QImage, QPixmap


class FrameEmitter(QThread):
    # establish signals from the frame that will be displayed in real time
    # within the GUI
    ImageBroadcast = pyqtSignal(QPixmap)
    FPSBroadcast = pyqtSignal(float)
    GridCountBroadcast = pyqtSignal(int)

    def __init__(self, monocalibrator, pixmap_edge_length=None):
        # pixmap_edge length is from the display window. Keep the display area
        # square to keep life simple.
        super(FrameEmitter, self).__init__()
        self.monocalibrator = monocalibrator
        self.pixmap_edge_length = pixmap_edge_length
        self.rotation_count = monocalibrator.camera.rotation_count
        self.undistort = False

    def run(self):
        self.ThreadActive = True

        while self.ThreadActive:
            # Grab a frame from the queue and broadcast to displays
            self.monocalibrator.grid_frame_ready_q.get()

            self.frame = self.monocalibrator.grid_frame
            self.apply_undistortion()
            self.apply_rotation()

            image = self.cv2_to_qlabel(self.frame)
            pixmap = QPixmap.fromImage(image)

            if self.pixmap_edge_length:
                pixmap = pixmap.scaled(
                    int(self.pixmap_edge_length),
                    int(self.pixmap_edge_length),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            self.ImageBroadcast.emit(pixmap)
            self.FPSBroadcast.emit(self.monocalibrator.stream.FPS_actual)
            self.GridCountBroadcast.emit(self.monocalibrator.grid_count)

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

    def apply_rotation(self):
        # logging.debug("Applying Rotation")
        if self.monocalibrator.camera.rotation_count == 0:
            pass
        elif self.monocalibrator.camera.rotation_count in [1, -3]:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.monocalibrator.camera.rotation_count in [2, -2]:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)
        elif self.monocalibrator.camera.rotation_count in [-1, 3]:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def apply_undistortion(self):

        if self.undistort == True:  # and self.mono_cal.is_calibrated:
            self.frame = cv2.undistort(
                self.frame,
                self.monocalibrator.camera.camera_matrix,
                self.monocalibrator.camera.distortion,
            )


if __name__ == "__main__":
    pass

    # not much to look at here... go to camera_config_dialogue.py for test
