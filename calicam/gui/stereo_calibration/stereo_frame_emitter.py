import logging

LOG_FILE = "log/stereo_frame_emitter.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from datetime import datetime
from pathlib import Path
from threading import Event

import cv2
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QImage, QPixmap

# from calicam.calibration.stereo_frame_builder import StereoFrameBuilder


class StereoFrameEmitter(QThread):
    # establish signals from the frame that will be displayed in real time
    # within the GUI
    StereoFramesBroadcast = pyqtSignal(object)
    StereoCalOutBroadcast = pyqtSignal(object)
    # GridCountBroadcast = pyqtSignal(int)

    def __init__(self, stereo_frame_builder):
        super(StereoFrameEmitter, self).__init__()
        self.stereo_frame_builder = stereo_frame_builder
        self.stop_event = Event()
        self.stereo_outputs = self.stereo_frame_builder.stereo_calibrator.stereo_outputs

    def run(self):
        self.ThreadActive = True

        while not self.stop_event.is_set():
            # wait for newly processed frames to become available
            self.stereo_frame_builder.set_current_synched_frames()

            frame_dict = self.stereo_frame_builder.get_stereoframe_pairs()

            # convert cv2 frames to pixmap for dialog
            for pair, frame in frame_dict.items():
                image = self.cv2_to_qlabel(frame)  # convert to qlabel
                pixmap = QPixmap.fromImage(image)  # and then to pixmap
                frame_dict[pair] = pixmap

            self.StereoFramesBroadcast.emit(frame_dict)
            self.StereoCalOutBroadcast.emit(self.stereo_outputs)
            logging.debug(f"stereo output dictionary: {self.stereo_outputs}")
        logging.info("Stereoframe emitter successfully shutdown...")

    def stop(self):
        self.stop_event.set()
        self.quit()

    def cv2_to_qlabel(self, frame):
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        qt_frame = QImage(
            Image.data,
            Image.shape[1],
            Image.shape[0],
            QImage.Format.Format_RGB888,
        )
        return qt_frame


if __name__ == "__main__":
    pass

    # not much to assess here, go to stereo_dialog
