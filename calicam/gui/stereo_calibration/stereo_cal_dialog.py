import logging

LOG_FILE = "log\stereo_cal_dialog.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
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
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from calicam.gui.stereo_calibration.stereo_pair_widget import StereoPairWidget


class StereoCalDialog(QScrollArea):
    def __init__(self, session):
        super(StereoCalDialog, self).__init__()
        self.session = session
        self.stereo_frame_emitter = session.stereo_frame_emitter
        self.vbox = QVBoxLayout()
        self.setLayout(self.vbox)

        self.build_top_controls()
        self.vbox.addLayout(self.top_controls)

        for pair in session.stereocalibrator.pairs:
            pair_widget = StereoPairWidget(self.session, self.stereo_frame_emitter, pair)
            self.vbox.addWidget(pair_widget)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

    def build_top_controls(self):

        self.top_controls = QHBoxLayout()
        self.top_controls.setContentsMargins(20, 0, 20, 0)

        # add a spin box to control the frame rate
        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setValue(6)

        self.session.stereocalibrator.synchronizer.fps_target = (
            self.frame_rate_spin.value()
        )

        def on_frame_rate_spin(fps_rate):
            self.session.stereocalibrator.synchronizer.update_fps_targets(fps_rate)

        self.frame_rate_spin.valueChanged.connect(on_frame_rate_spin)

        self.top_controls.addWidget(QLabel("FPS:"))
        self.top_controls.addWidget(self.frame_rate_spin)

        # and a spin box to control how many captures needed to calibrate
        self.capture_count_spin = QSpinBox()
        self.capture_count_spin.setValue(5)
        # initialize value
        self.session.stereocalibrator.grid_count_trigger = (
            self.capture_count_spin.value()
        )

        def on_capture_count_spin(capture_count):
            self.session.stereocalibrator.grid_count_trigger = capture_count

        self.capture_count_spin.valueChanged.connect(on_capture_count_spin)
        self.top_controls.addWidget(QLabel("Capture Ceiling:"))
        self.top_controls.addWidget(self.capture_count_spin)

        # and a spin box to control the wait time between captures
        self.wait_time_spin = QDoubleSpinBox()
        self.wait_time_spin.setValue(self.session.stereocalibrator.wait_time)
        self.wait_time_spin.setSingleStep(0.1)
        # self.session.stereocalibrator.wait_time = self.wait_time_spin.value()

        def on_wait_time_spin(wait_time):
            self.session.stereocalibrator.wait_time = wait_time

        self.wait_time_spin.valueChanged.connect(on_wait_time_spin)
        self.top_controls.addWidget(QLabel("Time Between Capture:"))
        self.top_controls.addWidget(self.wait_time_spin)

        # add button to save params to config file
        self.save_btn = QPushButton("Save Calibration")

        def on_save_btn(click):
            logging.debug("Saving stereocalibration")
            self.session.save_stereocalibration()

        self.save_btn.clicked.connect(on_save_btn)
        self.top_controls.addWidget(self.save_btn)


if __name__ == "__main__":
    from time import time

    from calicam.gui.stereo_calibration.stereo_frame_emitter import StereoFrameEmitter
    from calicam.session import Session

    repo = Path(__file__).parent.parent.parent.parent
    config_path = Path(repo, "sessions", "high_res_session")
    session = Session(config_path)
    session.load_cameras()
    session.load_streams()
    session.adjust_resolutions()
    session.load_stereo_tools()

    logging.info("Creating Camera Config Dialog")

    App = QApplication(sys.argv)
    stereo_frame_emitter = StereoFrameEmitter(session.stereo_frame_builder)
    stereo_frame_emitter.start()

    # cam_dialog = StereoPairWidget(session, test_pair)

    stereo_cal = StereoCalDialog(session)
    logging.info("About to show camera config dialog")
    stereo_cal.show()

    sys.exit(App.exec())
