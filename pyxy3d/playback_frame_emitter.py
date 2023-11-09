
import pyxy3d.logger

from datetime import datetime
from pathlib import Path
from time import sleep
from threading import Event
from queue import Queue

import cv2
from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QImage, QPixmap
from pyxy3d.calibration.monocalibrator import MonoCalibrator
from pyxy3d.calibration.intrinsic_calibrator import IntrinsicCalibrator
from pyxy3d.recording.recorded_stream import RecordedStream

logger = pyxy3d.logger.get(__name__)

class PlaybackFrameEmitter(QThread):
    # establish signals that will be displayed within the GUI
    ImageBroadcast = Signal(QPixmap)
    GridCountBroadcast = Signal(int)

    def __init__(self, recorded_stream:RecordedStream, pixmap_edge_length=500):
        # pixmap_edge length is from the display window. Keep the display area
        # square to keep life simple.
        super(PlaybackFrameEmitter, self).__init__()
        self.stream = recorded_stream
        self.frame_packet_q = Queue()
        self.stream.subscribe(self.frame_packet_q)
        self.pixmap_edge_length = pixmap_edge_length
        self.undistort = False
        self.keep_collecting = Event()
         
    def run(self):
        self.keep_collecting.set()

        while self.keep_collecting.is_set():
            # Grab a frame from the queue and broadcast to displays
            # self.monocalibrator.grid_frame_ready_q.get()
            logger.info("Getting frame packet from queue")
            frame_packet = self.frame_packet_q.get()
            self.frame = frame_packet.frame_with_points

            # self.apply_undistortion()
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
            # self.GridCountBroadcast.emit(self.monocalibrator.grid_count)

        logger.info(f"Thread loop within frame emitter at port {self.stream.port} successfully ended")

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
        if self.stream.rotation_count == 0:
            pass
        elif self.stream.rotation_count in [1, -3]:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.stream.rotation_count in [2, -2]:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)
        elif self.stream.rotation_count in [-1, 3]:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def apply_undistortion(self):
        
        if self.undistort:  # and self.mono_cal.is_calibrated:
            self.frame = cv2.undistort(
                self.frame,
                self.stream.camera.matrix,
                self.stream.camera.distortions,
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
