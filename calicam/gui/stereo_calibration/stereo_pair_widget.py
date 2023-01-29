# prior to tackling this, you need to create the stereo frame emitter,
# then the widget for a single pair before rolling them up into one
# larger set of dialogs

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

from calicam.gui.stereo_calibration.stereo_frame_emitter import StereoFrameEmitter


class StereoPairWidget(QWidget):
    def __init__(self, session, stereo_frame_emitter, pair):
        super(StereoPairWidget, self).__init__()
        self.session = session
        self.stereo_frame_emitter = stereo_frame_emitter

        self.pair = pair

        self.setWindowTitle("Stereocalibration")

        self.build_frame_pair_group()

        ######## Primarily horizontal layout
        self.hbox = QHBoxLayout(self)
        # self.hbox.setContentsMargins(0, 0, 0, 0)
        self.hbox.addWidget(self.frame_pair_group)

    def build_frame_pair_group(self):
        # outline group box to contain main elements
        self.frame_pair_group = QGroupBox(f"Cameras {self.pair[0]} and {self.pair[1]}")
        self.frame_pair_group.setContentsMargins(0, 0, 0, 0)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(20, 20, 20, 20)
        self.frame_pair_group.setLayout(hbox)

        ## construct Frame from qlabel
        self.frame_display = QLabel()

        ### hook up frame signal and slot
        def ImageUpdateSlot(stereoframes):
            pixmap = stereoframes[self.pair]
            self.frame_display.setPixmap(pixmap)

        self.stereo_frame_emitter.StereoFramesBroadcast.connect(ImageUpdateSlot)

        ### add frame to hbox (left most element)
        hbox.addWidget(self.frame_display)

        ## construct vbox for grid count / RMSE / reset button
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        ### create text labels
        self.grid_count = QLabel()
        self.error_reprojection = QLabel()

        ### hook up to incoming signal of stereo dict
        def StereoCalOutUpdateSlot(stereocal_output):
            count = str(stereocal_output[self.pair]["grid_count"])

            # make sure that RMSE is rounded, dealing with possibility of None
            error_reprojection = stereocal_output[self.pair]["RMSE"]
            if error_reprojection is not None:
                error_reprojection = round(error_reprojection, 2)

            error_reprojection = str(error_reprojection)

            self.grid_count.setText(f"Captured Boards: {count}")
            self.error_reprojection.setText(f"RMSE: {error_reprojection}")

        self.stereo_frame_emitter.StereoCalOutBroadcast.connect(StereoCalOutUpdateSlot)

        ### create button to reset stereo calibration for this pair
        self.reset_btn = QPushButton("Reset")

        def reset_stereo_cal():
            logging.debug(f"Resetting stereocal data associated with pair {self.pair}")
            self.session.stereocalibrator.reset_pair(self.pair)

        self.reset_btn.clicked.connect(reset_stereo_cal)

        # place items in vbox
        vbox.addWidget(self.grid_count)
        vbox.addWidget(self.error_reprojection)
        vbox.addWidget(self.reset_btn)

        ### add vbox to hbox (right most element)
        hbox.addLayout(vbox)


if __name__ == "__main__":
    from calicam.session import Session

    App = QApplication(sys.argv)

    repo = Path(__file__).parent.parent.parent.parent

    config_path = Path(repo, "sessions", "default_res_session")
    print(config_path)
    session = Session(config_path)
    session.load_cameras()
    session.load_streams()
    # session.adjust_resolutions()
    session.load_stereo_tools()

    logging.info("Creating Camera Config Dialog")
    test_pair = (0, 1)
    stereo_frame_emitter = StereoFrameEmitter(session.stereo_frame_builder)
    stereo_frame_emitter.start()
    cam_dialog = StereoPairWidget(session, stereo_frame_emitter, test_pair)

    logging.info("About to show camera config dialog")
    cam_dialog.show()

    sys.exit(App.exec())
