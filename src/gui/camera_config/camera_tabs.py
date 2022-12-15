import logging

LOG_FILE = "log/camera_tabs.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import sys
from pathlib import Path
from threading import Thread

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QTabWidget,
    QWidget,
    QSpinBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
)

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
# from src.gui.camera_config.frame_emitter import FrameEmitter
from src.gui.camera_config.camera_config_dialogue import CameraConfigDialog

from src.session import Session

class CameraTabs(QTabWidget):
    
    def __init__(self, session):
        super(CameraTabs, self).__init__()
        self.session = session

        self.setTabPosition(QTabWidget.TabPosition.North)
        self.add_cam_tabs()

    def add_cam_tabs(self):
        tab_names = [self.tabText(i) for i in range(self.count())]
        logging.info(f"Current tabs are: {tab_names}")

        if len(self.session.streams) > 0:
            for port, stream in self.session.streams.items():
                tab_name = f"Camera {port}"
                logging.info(f"Potentially adding {tab_name}")
                if tab_name in tab_names:
                    pass  # already here, don't bother
                else:
                    cam_tab = CameraConfigDialog(self.session, port)

                    # def on_save_click():
                    #     self.summary.camera_table.update_data()

                    # cam_tab.save_cal_btn.clicked.connect(on_save_click)

                    self.insertTab(port, cam_tab, tab_name)
                    # cam_tab.save_cal_btn.clicked.connect(self.summary.camera_table.update_data)
        else:
            logging.info("No cameras available")


if __name__ == "__main__":
    App = QApplication(sys.argv)

    repo = Path(__file__).parent.parent.parent.parent
    config_path = Path(repo, "sessions", "high_res_session")
    print(config_path)
    session = Session(config_path)
    session.load_cameras()
    session.load_stream_tools()
    # session.adjust_resolutions()
    session.load_monocalibrators()

    test_port = 0

    # cam_dialog = CameraConfigDialog(session, test_port)
    cam_tabs = CameraTabs(session)
    
    cam_tabs.show()

    sys.exit(App.exec())

