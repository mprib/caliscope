import logging

LOG_FILE = "log\capture_volume_dialog.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)


import sys
from pathlib import Path
from threading import Thread

import cv2
from PyQt6.QtCore import Qt
import pyqtgraph as pg
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
from calicam.gui.capture_volume.visualizer import CaptureVolumeVisualizer

class CaptureVolumeDialog(QWidget):
    def __init__(self, CaptureVolumeVisualizer):
        super(CaptureVolumeDialog, self).__init__()
        self.visualizer = CaptureVolumeVisualizer
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("This is a test"))
        self.layout().addWidget(self.visualizer.scene)
        self.setMinimumSize(500,500)
        self.visualizer.begin()
        
if __name__ == "__main__":

    from calicam.recording.recorded_stream import RecordedStreamPool
    from calicam.cameras.synchronizer import Synchronizer
    from calicam.calibration.charuco import Charuco
    from calicam.calibration.corner_tracker import CornerTracker
    from calicam.triangulate.paired_point_stream import PairedPointStream
    from calicam.triangulate.stereo_triangulator import StereoTriangulator
    from calicam.triangulate.stereo_triangulator import StereoTriangulator

    # set the location for the sample data used for testing
    repo = Path(__file__).parent.parent.parent.parent
    session_directory =Path(repo, "sessions", "high_res_session")
    # create playback streams to provide to synchronizer
    ports = [0, 2]
    recorded_stream_pool = RecordedStreamPool(ports, session_directory)
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
    recorded_stream_pool.play_videos()
    # create a corner tracker to locate board corners
    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )
    trackr = CornerTracker(charuco)

    # create a commmon point finder to grab charuco corners shared between the pair of ports
    pairs = [(ports[0], ports[1])]
    point_stream = PairedPointStream(
        synchronizer=syncr,
        pairs=pairs,
        tracker=trackr,

    )


    config_path = str(Path(session_directory, "config.toml"))
    triangulatr = StereoTriangulator(point_stream, config_path)

    app = QApplication(sys.argv)

    vizr = CaptureVolumeVisualizer(triangulatr)
    vizr.add_point_q(triangulatr.out_q)

    vizr_dialog = CaptureVolumeDialog(vizr)
    vizr_dialog.show()

    app.exec()