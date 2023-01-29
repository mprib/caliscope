# Built following the tutorials that begin here:
# https://www.pythonguis.com/tutorials/pyqt6-creating-your-first-window/

import logging
import sys

LOG_FILE = "log\main.log"
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
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from camera_table import CameraTable

from gui.camera_config.camera_config_dialogue import CameraConfigDialog
from calicam.gui.charuco_builder import CharucoBuilder
from calicam.session import Session
from calicam.gui.stereo_cal_dialog import StereoCalDialog
from calicam.gui.stereo_frame_emitter import StereoFrameEmitter


class MainWindow(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session
        app = QApplication.instance()
        screen = app.primaryScreen()
        DISPLAY_WIDTH = screen.size().width()
        DISPLAY_HEIGHT = screen.size().height()
        self.setMinimumSize(DISPLAY_WIDTH * 0.30, DISPLAY_HEIGHT * 0.7)
        self.cams_in_process = False
        self.setWindowTitle("FreeMocap Camera Calibration")
        self.setWindowIcon(QIcon("src/gui/icons/fmc_logo.ico"))

        #
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(True)
        self.setCentralWidget(self.tabs)

        self.summary = SessionSummary(self.session)
        self.tabs.addTab(self.summary, "&Summary")

        self.summary.launch_charuco_builder_btn.clicked.connect(
            self.launch_charuco_builder
        )

        self.summary.open_cameras_btn.clicked.connect(self.open_cams)
        self.summary.connect_cams_btn.clicked.connect(self.connect_cams)
        self.summary.find_cams_btn.clicked.connect(self.find_additional_cams)
        self.summary.launch_stereo_cal_btn.clicked.connect(self.launch_stereo_cal)

    def open_cams(self):

        tab_names = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        logging.debug(f"Current tabs are: {tab_names}")

        if len(self.session.streams) > 0:
            for port, stream in self.session.streams.items():
                tab_name = f"Camera {port}"
                logging.debug(f"Potentially adding {tab_name}")
                if tab_name in tab_names:
                    pass  # already here, don't bother
                else:
                    cam_tab = CameraConfigDialog(self.session, port)

                    def on_save_click():
                        self.summary.camera_table.update_data()

                    cam_tab.save_cal_btn.clicked.connect(on_save_click)

                    self.tabs.addTab(cam_tab, tab_name)
                    # cam_tab.save_cal_btn.clicked.connect(self.summary.camera_table.update_data)
        else:
            logging.info("No cameras available")

    def update_summary_image(self):
        self.summary.update_charuco_summary()

    def launch_charuco_builder(self):
        for t in range(0, self.tabs.count()):
            if self.tabs.tabText(t) == "Charuco Builder":
                return

        self.charuco_builder = CharucoBuilder(self.session)
        self.charuco_builder.save_btn.clicked.connect(self.update_summary_image)
        self.tabs.addTab(self.charuco_builder, "Charuco Builder")
        # self.tabs["Charuco Builder"].setClosable(True)

    def launch_stereo_cal(self):
        self.session.load_stereo_tools()
        self.stereo_frame_emitter = StereoFrameEmitter(
            self.session.stereo_frame_builder
        )
        self.stereo_frame_emitter.start()
        self.stereocal_dialog = StereoCalDialog(self.session, self.stereo_frame_emitter)
        self.tabs.addTab(self.stereocal_dialog, "Stereocalibration")

    def find_additional_cams(self):
        def find_cam_worker():

            self.session.find_additional_cameras()
            logging.info("Loading streams")
            self.session.load_stream_tools()
            logging.info("Loading monocalibrators")
            self.session.load_monocalibrators()
            logging.info("Adjusting resolutions")
            self.session.adjust_resolutions()
            logging.info("Updating Camera Table")
            self.summary.camera_table.update_data()

            self.summary.open_cameras_btn.click()

            self.cams_in_process = False
            self.summary.connect_cams_btn.setEnabled(True)
            
        if not self.cams_in_process:
            print("Searching for additional cameras...This may take a moment.")
            self.find = Thread(target=find_cam_worker, args=(), daemon=True)
            self.find.start()
        else:
            print("Cameras already connected or in process.")

    def connect_cams(self):
        def connect_cam_worker():
            self.cams_in_process = True
            logging.info("Loading Cameras")
            self.session.load_cameras()
            logging.info("Loading streams")
            self.session.load_stream_tools()
            logging.info("Adjusting resolutions")
            self.session.adjust_resolutions()
            logging.info("Loading monocalibrators")
            self.session.load_monocalibrators()
            logging.info("Updating Camera Table")
            self.summary.camera_table.update_data()

            # trying to call open_cams() directly created a weird bug that
            # may be due to all the different threads. This seemed to kick it
            # back to the main thread...
            self.summary.open_cameras_btn.click()

            self.cams_in_process = False

        if not self.cams_in_process:
            print("Connecting to cameras...This may take a moment.")
            self.connect = Thread(target=connect_cam_worker, args=(), daemon=True)
            self.connect.start()
        else:
            print("Cameras already connected or in process.")


class SessionSummary(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session

        # self.cams_connected = False

        self.scroll = (
            QScrollArea()
        )  # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()  # Widget that contains the collection of Vertical Box
        self.vbox = (
            QVBoxLayout()
        )  # The Vertical Box that contains the Horizontal Boxes of  labels and buttons
        # Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)

        # self.setGeometry(600, 100, 1000, 900)
        self.setWindowTitle("Scroll Area Demonstration")
        self.show()

        self.widget.setLayout(self.vbox)

        self.top_hbox = QHBoxLayout()
        self.vbox.addLayout(self.top_hbox)
        self.vbox.setAlignment(self.top_hbox, Qt.AlignmentFlag.AlignTop)

        self.charuco_group = QGroupBox("Charuco Board")
        self.charuco_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        # self.charuco_group.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.top_hbox.addWidget(self.charuco_group)

        self.cam_summary = QGroupBox("Single Camera Calibration")
        self.top_hbox.addWidget(self.cam_summary)
        self.top_hbox.setAlignment(self.cam_summary, Qt.AlignmentFlag.AlignTop)

        self.build_charuco_summary()
        self.build_cam_summary()
        self.build_stereo_summary()

    def build_charuco_summary(self):
        vbox = QVBoxLayout()
        self.charuco_group.setLayout(vbox)

        self.charuco_display = QLabel()
        self.charuco_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.charuco_display.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum
        )

        hbox = QHBoxLayout()
        vbox.addLayout(hbox)
        hbox.addWidget(self.charuco_display)

        self.charuco_summary = QLabel()
        hbox.addWidget(self.charuco_summary)
        hbox.setAlignment(self.charuco_display, Qt.AlignmentFlag.AlignBaseline)
        hbox.setAlignment(self.charuco_summary, Qt.AlignmentFlag.AlignBaseline)

        self.launch_charuco_builder_btn = QPushButton("&Launch Builder")
        self.launch_charuco_builder_btn.setMaximumSize(150, 30)
        vbox.addWidget(self.launch_charuco_builder_btn)
        vbox.setAlignment(
            self.launch_charuco_builder_btn, Qt.AlignmentFlag.AlignHCenter
        )

        self.update_charuco_summary()

    def update_charuco_summary(self):
        charuco_width = 200
        charuco_height = 200
        charuco_img = self.session.charuco.board_pixmap(charuco_width, charuco_height)
        self.charuco_display.setPixmap(charuco_img)
        self.charuco_summary.setText(self.session.charuco.summary())

    def build_cam_summary(self):
        self.cam_hbox = QHBoxLayout()
        self.cam_summary.setLayout(self.cam_hbox)

        left_vbox = QVBoxLayout()

        self.camera_table = CameraTable(self.session)
        self.camera_table.setFixedSize(250, 150)
        left_vbox.addWidget(self.camera_table)
        self.connect_cams_btn = QPushButton("&Connect to Cameras")

        if self.session.camera_count() == 0:
            self.connect_cams_btn.setEnabled(False)
            
        left_vbox.addWidget(self.connect_cams_btn)
        self.find_cams_btn = QPushButton("&Find Additional Cameras")
        left_vbox.addWidget(self.find_cams_btn)

        self.open_cameras_btn = QPushButton("Open Cameras")  # this button is invisible

        self.cam_hbox.addLayout(left_vbox)

    def build_stereo_summary(self):
        stereo_summary = QGroupBox("Stereocalibration")
        self.vbox.addWidget(stereo_summary)
        self.launch_stereo_cal_btn = QPushButton("Launch Stereocalibrator")
        stereo_vbox = QVBoxLayout()
        stereo_summary.setLayout(stereo_vbox)
        stereo_vbox.addWidget(self.launch_stereo_cal_btn)


if __name__ == "__main__":
    repo = Path(__file__).parent.parent.parent
    # config_path = Path(repo, "sessions", "default_res_session")
    config_path = Path(repo, "sessions", "high_res_session")
    print(config_path)
    session = Session(config_path)

    # comment out this next line if you want to save cameras after closing
    # session.delete_all_cam_data() 

    app = QApplication(sys.argv)

    window = MainWindow(session)
    # window = SessionSummary(session)
    window.show()

    app.exec()
