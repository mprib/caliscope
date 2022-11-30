import logging

LOG_FILE = "log/stereo_dialog.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import sys
from pathlib import Path
from threading import Thread

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from frame_emitter import FrameEmitter

from src.calibration.charuco import Charuco
from src.cameras.camera import Camera
from src.cameras.video_stream import VideoStream
from src.session import Session


class StereoPairConfig(QWidget):
    def __init__(self, session, pair):

        self.frame_emitter = FrameEmitter()


if __name__ == "__main__":
    App = QApplication(sys.argv)

    repo = Path(__file__).parent.parent.parent
    config_path = Path(repo, "default_session")
    print(config_path)
    session = Session(config_path)
    session.load_cameras()
    session.load_stream_tools()
    session.load_monocalibrators()

    test_port = 0

    logging.info("Creating Camera Config Dialog")
    cam_dialog = StereoPairConfig(session, test_port)

    logging.info("About to show camera config dialog")
    cam_dialog.show()

    sys.exit(App.exec())
