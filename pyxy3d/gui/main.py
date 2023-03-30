import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import sys
import shutil
import time
from pathlib import Path
from threading import Thread
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import (
    QWidget,
    QApplication,
    QVBoxLayout,
    QHBoxLayout,
    QDockWidget,
    QFileDialog,
    QStackedWidget,
)

from pyxy3d.session import Session, Stage
from pyxy3d.gui.wizard_charuco import WizardCharuco
from pyxy3d.gui.camera_config.camera_tabs import CameraWizard
from pyxy3d.gui.wizard_directory import WizardDirectory
from pyxy3d import __root__, __app_dir__
from pyxy3d.session import Stage
from pyxy3d.gui.qt_logger import QtLogger
from pyxy3d.gui.stereoframe.stereo_frame_widget import StereoFrameWidget
from pyxy3d.gui.vizualize.capture_volume_widget import CaptureVolumeWidget


class CalibrationWizard(QStackedWidget):
    cameras_connected = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.CAMS_IN_PROCESS = False

        self.setWindowTitle("Camera Calibration Wizard")
        self.setWindowIcon(QIcon(str(Path(__root__, "pyxy3d/gui/icons/orb.svg"))))
        self.wizard_directory = WizardDirectory()
        self.addWidget(self.wizard_directory)  # index:1
        self.setCurrentIndex(0)
        self.connect_widgets()

    def connect_widgets(self):
        self.wizard_directory.launch_wizard_btn.clicked.connect(self.begin_wizard)
        self.cameras_connected.connect(self.on_cameras_connect)

    # create first page of wizard (charuco builder)
    def begin_wizard(self):
        if hasattr(self, "wizard_charuco"):
            self.setCurrentIndex(1)
        else:
            logger.info("Launching session")
            self.launch_session()
            logger.info("Creating charuco wizard session")
            self.wizard_charuco = WizardCharuco(self.session)
            self.wizard_charuco.navigation_bar.next_wizard_step_btn.clicked.connect(
                self.next_to_camera_config_wizard
            )
            logger.info("Adding charuco wizard")
            self.addWidget(self.wizard_charuco)
            logger.info("Setting index to 2 to activate widget")
            self.setCurrentIndex(1)

    # Start Session
    def launch_session(self):
        if self.wizard_directory.create_new_radio.isChecked():
            # only need to create a new session in the given directory:
            self.session_directory = self.wizard_directory.new_path.textbox.text()
            self.session = Session(self.session_directory)
        else:
            # need to copy over config from old directory to new directory before launching
            self.session_directory = self.wizard_directory.modified_path.textbox.text()
            old_config_path = self.wizard_directory.original_path.textbox.text()

            ## but check if it's the same directory
            if self.session_directory == old_config_path:
                # in which case don't do anything
                pass
            else:
                shutil.copyfile(
                    str(Path(old_config_path, "config.toml")),
                    str(Path(self.session_directory, "config.toml")),
                )

            self.session = Session(self.session_directory)

    ######################## STEP 1: Charuco Builder ###########################

    def next_to_camera_config_wizard(self):
        if hasattr(self, "camera_wizard"):
            logger.info("Camera wizard already exists; changing stack current index")
            self.setCurrentIndex(2)
            active_port = self.camera_wizard.camera_tabs.currentIndex()
            self.camera_wizard.camera_tabs.toggle_tracking(active_port)
            logger.info("updating charuco in case necessary")
            for port, stream in self.session.streams.items():
                stream.update_charuco(self.session.charuco)
        else:
            logger.info("Initiating Camera Connection")
            self.initiate_camera_connection()
            self.qt_logger = QtLogger("Connecting to Cameras...")
            self.qt_logger.show()

    def initiate_camera_connection(self):

        if len(self.session.cameras) > 0:
            logger.info("Cameras already connected")
        else:

            def connect_to_cams_worker():
                self.CAMS_IN_PROCESS = True
                logger.info("Initiating camera connect worker")

                # find out if you are loading cameras or finding cameras
                if self.session.get_configured_camera_count() > 0:
                    self.session.load_cameras()
                    logger.info("Camera connect worker about to load stream tools")
                    self.session.load_streams()
                else:
                    # I believe find_cameras will establish the streams as well...
                    self.session.find_cameras()
                logger.info("Camera connect worker about to adjust resolutions")
                self.session.adjust_resolutions()
                logger.info("Camera connect worker about to load monocalibrators")
                self.session.load_monocalibrators()
                self.CAMS_IN_PROCESS = False

                logger.info("emitting cameras_connected signal")
                self.cameras_connected.emit()
                if hasattr(self, "qt_logger"):
                    del self.qt_logger
                # self.qt_logger.hide()

        if self.CAMS_IN_PROCESS:
            logger.info("Already attempting to connect to cameras...")
        else:
            self.connect_cams = Thread(
                target=connect_to_cams_worker, args=[], daemon=True
            )
            self.connect_cams.start()

    def on_cameras_connect(self):
        # load cameras wizard once the cameras are actually connected
        self.camera_wizard = CameraWizard(self.session)
        self.addWidget(self.camera_wizard)
        self.setCurrentIndex(2)
        self.camera_wizard.navigation_bar.back_btn.clicked.connect(
            self.back_to_charuco_wizard
        )
        self.camera_wizard.navigation_bar.next_btn.clicked.connect(
            self.next_to_stereoframe
        )

    ####################### STEP 2: Single Camera Calibration #################
    def back_to_charuco_wizard(self):
        self.setCurrentIndex(1)
        self.session.pause_all_monocalibrators()

    def next_to_stereoframe(self):
        if hasattr(self, "stereoframe"):
            self.session.unpause_synchronizer()
        else:
            self.stereoframe = StereoFrameWidget(self.session)
            self.addWidget(self.stereoframe)
            self.stereoframe.navigation_bar.back_btn.clicked.connect(
                self.back_to_camera_config_wizard
            )
            self.stereoframe.calibration_complete.connect(self.launch_capture_volume)
        self.setCurrentIndex(3)
        self.session.pause_all_monocalibrators()

    ###################### Stereocalibration  ######################################

    def back_to_camera_config_wizard(self):
        # from stereoframe to camera config
        self.setCurrentIndex(2)
        active_port = self.camera_wizard.camera_tabs.currentIndex()
        self.camera_wizard.camera_tabs.toggle_tracking(active_port)
        self.session.pause_synchronizer()

    def launch_capture_volume(self):
        logger.info("Creating Capture Volume widget")
        self.capture_volume = CaptureVolumeWidget(self.session)
        logger.info("Adding capture volume widget to main Wizard")
        self.addWidget(self.capture_volume)
        logger.info("Set current index to capture volume widget")
        self.setCurrentIndex(4)

        # self.launch_cv_thread = Thread(target=worker, args=(), daemon=True)
        # self.launch_cv_thread.start()


def launch_pyxy3d():

    app = QApplication(sys.argv)
    window = CalibrationWizard()
    window.show()

    app.exec()


if __name__ == "__main__":

    launch_pyxy3d()
