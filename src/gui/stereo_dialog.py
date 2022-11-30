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

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.gui.stereo_frame_emitter import StereoFrameEmitter


class StereoPairConfigDialog(QDialog):
    def __init__(self, session, pair):
        super(StereoPairConfigDialog, self).__init__()

        self.stereo_frame_emitter = StereoFrameEmitter(session.stereo_frame_builder)
        self.stereo_frame_emitter.start()

        self.pair = pair

        # get size of display for reference
        App = QApplication.instance()
        DISPLAY_WIDTH = App.primaryScreen().size().width()
        DISPLAY_HEIGHT = App.primaryScreen().size().height()

        self.setWindowTitle("Stereocalibration")

        self.build_frame_display()

        ######## Primarily horizontal layout
        self.hbox = QHBoxLayout(self)
        self.hbox.addWidget(self.frame_display)
        self.hbox.addWidget(self.grid_count)

    def build_frame_display(self):
        # return a QLabel that is linked to the constantly changing image

        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_display.setFixedWidth(self.width())
        self.frame_display.setFixedHeight(self.height())

        def ImageUpdateSlot(stereoframes):
            pixmap = stereoframes[self.pair]
            self.frame_display.setPixmap(pixmap)

        self.stereo_frame_emitter.StereoFramesBroadcast.connect(ImageUpdateSlot)

        self.grid_count = QLabel()

        def GridCountUpdateSlot(grid_counts):
            count = str(grid_counts[self.pair])
            self.grid_count.setText(f"Captured Boards: {count}")

        self.stereo_frame_emitter.GridCountBroadcast.connect(GridCountUpdateSlot)


if __name__ == "__main__":
    from src.session import Session

    App = QApplication(sys.argv)

    repo = Path(__file__).parent.parent.parent
    config_path = Path(repo, "sessions", "default_session")
    print(config_path)
    session = Session(config_path)
    session.load_cameras()
    session.load_stream_tools()
    session.load_monocalibrators()
    session.load_stereo_tools()

    logging.info("Creating Camera Config Dialog")
    test_pair = (0, 1)
    cam_dialog = StereoPairConfigDialog(session, test_pair)

    logging.info("About to show camera config dialog")
    cam_dialog.show()

    sys.exit(App.exec())
