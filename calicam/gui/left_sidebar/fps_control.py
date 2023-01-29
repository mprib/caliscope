
# Built following the tutorials that begin here:
# https://www.pythonguis.com/tutorials/pyqt6-creating-your-first-window/

import logging
import sys

LOG_FILE = r"log\fps_control.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import time
from pathlib import Path
from threading import Thread

from numpy import char
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from calicam.session import Session


class FPSControl(QWidget):
    def __init__(self, session, default_fps=6):
        super().__init__()
        self.session = session
        self.setLayout(QHBoxLayout())

        logging.debug("Building FPS Control")

        self.layout().addWidget(QLabel("Target:"))
        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setValue(default_fps)
        self.frame_rate_spin.setEnabled(False)  # start out this way..enable when synchronizer constructed
        self.layout().addWidget(self.frame_rate_spin)
        
        def on_frame_rate_spin(fps_rate):
            try:
                self.session.synchronizer.fps_target = fps_rate
                logging.info(f"Changing synchronizer frame rate")
            except(AttributeError):
                logging.warning("Unable to change synch fps...may need to load stream tools") 

        self.frame_rate_spin.valueChanged.connect(on_frame_rate_spin)

if __name__ == "__main__":
    repo = Path(__file__).parent.parent.parent.parent
    config_path = Path(repo, "sessions", "high_res_session")
    

    session = Session(config_path)
    session.load_cameras()
    session.load_streams()

    print(session.config)
    app = QApplication(sys.argv)
    fps_control = FPSControl(session)
    fps_control.show()
    sys.exit(app.exec())