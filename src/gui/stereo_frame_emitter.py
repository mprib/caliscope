import logging

LOG_FILE = "log/stereo_frame_emitter.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from datetime import datetime
from pathlib import Path

import cv2
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QImage, QPixmap

from src.calibration.stereo_frame_builder import StereoFrameBuilder


class StereoFrameEmitter(QThread):
    # establish signals from the frame that will be displayed in real time
    # within the GUI
    StereoFramesBroadcast = pyqtSignal(object)
    # GridCountBroadcast = pyqtSignal(int)

    def __init__(self, stereo_frame_builder):
        super(StereoFrameEmitter, self).__init__()
        self.stereo_frame_builder = stereo_frame_builder

    def run(self):
        self.ThreadActive = True

        while self.ThreadActive:
            # wait for newly processed frames to become available
            self.stereo_frame_builder.set_current_bundle()

            frame_dict = self.stereo_frame_builder.get_stereoframe_pairs()

            # convert cv2 frames to pixmap for dialog
            for pair, frame in frame_dict.items():
                image = self.cv2_to_qlabel(frame)  # convert to qlabel
                pixmap = QPixmap.fromImage(image)  # and then to pixmap
                frame_dict[pair] = pixmap

            # if self.pixmap_edge_length:
            #     pixmap = pixmap.scaled(
            #         self.pixmap_edge_length,
            #         self.pixmap_edge_length,
            #         Qt.AspectRatioMode.KeepAspectRatio,
            #     )
            self.StereoFramesBroadcast.emit(frame_dict)
            # self.GridCountBroadcast.emit(self.monocalibrator.grid_count)

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

    # time.sleep(3)


if __name__ == "__main__":
    pass

    # not much to assess here, go to stereo_dialog
