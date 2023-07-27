
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from datetime import datetime
from pathlib import Path
from time import sleep
from threading import Event

import cv2
from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QImage, QPixmap
from pyxy3d.calibration.monocalibrator import MonoCalibrator


class FrameEmitter(QThread):
    # establish signals from the frame that will be displayed in real time
    # within the GUI
    ImageBroadcast = Signal(QPixmap)
    FPSBroadcast = Signal(float)
    GridCountBroadcast = Signal(int)

    def __init__(self, monocalibrator:MonoCalibrator, pixmap_edge_length=None):
        # pixmap_edge length is from the display window. Keep the display area
        # square to keep life simple.
        super(FrameEmitter, self).__init__()
        self.monocalibrator = monocalibrator
        self.pixmap_edge_length = pixmap_edge_length
        self.rotation_count = monocalibrator.camera.rotation_count
        self.undistort = False
        self.keep_collecting = Event()
    def run(self):
        self.keep_collecting.set()

        while self.keep_collecting.is_set():
            # Grab a frame from the queue and broadcast to displays
            self.monocalibrator.grid_frame_ready_q.get()
            self.frame = self.monocalibrator.grid_frame

            self.apply_undistortion()
            self.frame = resize_to_square(self.frame)
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
            
            # moved to monocalibrator...delete if works well
            self.FPSBroadcast.emit(self.monocalibrator.stream.FPS_actual)
            self.GridCountBroadcast.emit(self.monocalibrator.grid_count)

        logger.info(f"Thread loop within frame emitter at port {self.monocalibrator.port} successfully ended")

    def stop(self):
        self.keep_collecting = False
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
        # logger.debug("Applying Rotation")
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
                self.monocalibrator.camera.matrix,
                self.monocalibrator.camera.distortions,
            )


def resize_to_square(frame):

    height = frame.shape[0]
    width = frame.shape[1]

    padded_size = max(height, width)

    height_pad = int((padded_size - height) / 2)
    width_pad = int((padded_size - width) / 2)
    pad_color = [0, 0, 0]

    frame = cv2.copyMakeBorder(
        frame,
        height_pad,
        height_pad,
        width_pad,
        width_pad,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    return frame

if __name__ == "__main__":
    pass

    # not much to look at here... go to camera_config_dialogue.py for test
