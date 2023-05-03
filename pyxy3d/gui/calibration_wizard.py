import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import os
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
from pyxy3d.trackers.charuco_tracker import CharucoTracker, CharucoTrackerFactory
from pyxy3d.gui.qt_logger import QtLogger
from pyxy3d.gui.stereoframe.stereo_frame_widget import (
    StereoFrameWidget,
    MIN_THRESHOLD_FOR_EARLY_CALIBRATE,
)
from pyxy3d.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget
from pyxy3d.configurator import Configurator


class CalibrationWizard(QStackedWidget):
    cameras_connected = pyqtSignal()

    def __init__(self, session_path:Path):
        super().__init__()
        self.CAMS_IN_PROCESS = False

        self.setWindowTitle("Camera Calibration Wizard")
        self.setWindowIcon(QIcon(str(Path(__root__, "pyxy3d/gui/icons/pyxy_logo.svg"))))
        self.session_path = session_path
        self.config = Configurator(self.session_path)
        self.session = Session(self.config)

        # self.launch_session()
        logger.info("Creating charuco wizard session")
        self.wizard_charuco = WizardCharuco(self.session)
        self.wizard_charuco.navigation_bar.next_wizard_step_btn.clicked.connect(
            self.next_to_camera_config
        )
        logger.info("Adding charuco wizard")
        self.addWidget(self.wizard_charuco)
        logger.info("Setting index to 2 to activate widget")
        self.setCurrentWidget(self.wizard_charuco)

        self.connect_widgets()

    def connect_widgets(self):
        # self.wizard_directory.launch_wizard_btn.clicked.connect(self.begin_wizard)
        self.cameras_connected.connect(self.on_cameras_connect)

    def launch_session(self):
        if self.wizard_directory.create_new_radio.isChecked():
            # only need to create a new session in the given directory:
            self.session_path = self.wizard_directory.new_path.textbox.text()
            configurator = Configurator(self.session_path)
            self.session = Session(configurator)
        else:
            # need to copy over config from old directory to new directory before launching
            self.session_path = self.wizard_directory.modified_path.textbox.text()
            old_config_path = self.wizard_directory.original_path.textbox.text()

            ## but check if it's the same directory
            if self.session_path == old_config_path:
                # in which case don't do anything
                pass
            else:
                shutil.copyfile(
                    str(Path(old_config_path, "config.toml")),
                    str(Path(self.session_path, "config.toml")),
                )

            configurator = Configurator(self.session_path)
            self.session = Session(configurator)

    ######################## STEP 1: Charuco Builder ###########################

    def next_to_camera_config(self):
        if hasattr(self, "camera_config"):
            logger.info("Camera config already exists; changing stack current index")
            self.setCurrentWidget(self.camera_config)
            active_port = self.camera_config.camera_tabs.currentIndex()
            self.camera_config.camera_tabs.toggle_tracking(active_port)
            logger.info("updating charuco in case necessary")
            charuco_tracker_factory = CharucoTrackerFactory(self.session.charuco)
            for port, stream in self.session.streams.items():
                stream.update_tracker(charuco_tracker_factory.get_tracker())
        else:
            logger.info("Initiating Camera Connection")
            self.initiate_camera_connection()
            self.qt_logger = QtLogger("Connecting to Cameras...")
            self.qt_logger.show()

    def initiate_camera_connection(self):
        if len(self.session.streams) > 0:
            logger.info("Cameras already connected")
        else:

            def connect_to_cams_worker():
                self.CAMS_IN_PROCESS = True
                logger.info("Initiating camera connect worker")

                # find out if you are loading cameras or finding cameras
                if self.session.get_configured_camera_count() > 0:
                    # self.# session.load_cameras()
                    logger.info("Camera connect worker about to load stream tools")

                    self.session.load_streams(
                        tracker_factory=CharucoTrackerFactory(self.session.charuco)
                    )
                else:
                    logger.info(
                        f"No previous configured cameras detected...searching for cameras...."
                    )
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
        # load cameras config once the cameras are actually connected
        self.camera_config = CameraWizard(self.session)
        self.addWidget(self.camera_config)
        self.setCurrentWidget(self.camera_config)
        self.camera_config.navigation_bar.back_btn.clicked.connect(
            self.back_to_charuco_wizard
        )
        self.camera_config.navigation_bar.next_btn.clicked.connect(
            self.next_to_stereoframe
        )

    ####################### STEP 2: Single Camera Calibration #################
    def back_to_charuco_wizard(self):
        self.setCurrentWidget(self.wizard_charuco)
        self.session.pause_all_monocalibrators()

    def next_to_stereoframe(self):
        self.session.pause_all_monocalibrators()

        if hasattr(self.session, "synchronizer"):
            self.session.unpause_synchronizer()

        self.launch_new_stereoframe()

    def launch_new_stereoframe(self):
        self.stereoframe = StereoFrameWidget(self.session)
        self.addWidget(self.stereoframe)
        self.setCurrentWidget(self.stereoframe)

        self.stereoframe.navigation_bar.back_btn.clicked.connect(
            self.back_to_camera_config_wizard
        )

        self.stereoframe.calibration_complete.connect(self.next_to_capture_volume)
        self.stereoframe.calibration_initiated.connect(self.show_calibration_qt_logger)
        self.stereoframe.terminate.connect(self.refresh_stereoframe)

    ###################### Stereocalibration  ######################################
    def refresh_stereoframe(self):
        logger.info("Set current widget to config temporarily")
        self.setCurrentWidget(self.camera_config)

        logger.info("Remove stereoframe")
        self.removeWidget(self.stereoframe)
        self.stereoframe.frame_builder.unsubscribe_from_synchronizer()
        del self.stereoframe

        logger.info("Create new stereoframe")
        self.launch_new_stereoframe()

    def show_calibration_qt_logger(self):
        """
        Calibration is initiated back on the stereoframe widget,here only
        the logger launch is managed because it is main that must delete the logger
        """
        logger.info("Launching calibration qt logger")
        self.qt_logger = QtLogger("Calibrating camera array...")
        self.qt_logger.show()

    def back_to_camera_config_wizard(self):
        logger.info("Moving back to camera config from stereoframe")
        self.setCurrentWidget(self.camera_config)
        self.session.pause_synchronizer()

        self.stereoframe.frame_builder.unsubscribe_from_synchronizer()
        self.removeWidget(self.stereoframe)
        del self.stereoframe

        active_port = self.camera_config.camera_tabs.currentIndex()
        self.camera_config.camera_tabs.toggle_tracking(active_port)

    def next_to_capture_volume(self):
        logger.info("Creating Capture Volume widget")
        self.capture_volume = CaptureVolumeWidget(self.session)
        logger.info("Adding capture volume widget to main Wizard")
        self.addWidget(self.capture_volume)
        logger.info("Set current index to capture volume widget")
        self.setCurrentWidget(self.capture_volume)
        self.capture_volume.navigation_bar.back_btn.clicked.connect(
            self.back_to_stereo_frame
        )

        del self.qt_logger

    ################## Capture Volume ########################
    def back_to_stereo_frame(self):
        logger.info("Set current widget to config temporarily")
        self.setCurrentWidget(self.camera_config)

        logger.info("Remove stereoframe")
        self.removeWidget(self.stereoframe)
        logger.info("Remove capture volume")
        self.removeWidget(self.capture_volume)
        del self.capture_volume
        self.stereoframe.frame_builder.unsubscribe_from_synchronizer()
        del self.stereoframe

        logger.info("Create new stereoframe")
        self.launch_new_stereoframe()
        self.session.unpause_synchronizer()


def launch_calibration_wizard(session_path:Path):
    app = QApplication(sys.argv)
    window = CalibrationWizard(session_path=session_path)
    window.show()
    app.exec()


if __name__ == "__main__":
    from pyxy3d import __root__ 
    session_path = Path(__root__, "dev", "sample_sessions", "296")

    launch_calibration_wizard(session_path)
    
    # app = QApplication(sys.argv)
    # window = CalibrationWizard(session_path)
    # window.show()

    # app.exec()
    
    