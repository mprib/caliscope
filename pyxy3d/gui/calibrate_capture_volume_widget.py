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

from pyxy3d.session.session import Session, SessionMode
from pyxy3d.gui.charuco_widget import CharucoWidget
from pyxy3d.gui.camera_config.intrinsic_calibration_widget import IntrinsicCalibrationWidget
from pyxy3d import __root__, __app_dir__
from pyxy3d.trackers.charuco_tracker import CharucoTracker
# from pyxy3d.gui.qt_logger import QtLogger
from pyxy3d.gui.extrinsic_calibration_widget import (
    ExtrinsicCalibrationWidget,
    MIN_THRESHOLD_FOR_EARLY_CALIBRATE,
)
from pyxy3d.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget
from pyxy3d.configurator import Configurator


class CalibrateCaptureVolumeWidget(QStackedWidget):
    """
    A combination of the Extrinsic Calibration Widget and the Capture Volume Widget
    Allows the user to move back and forth between the two functions.
    """
    cameras_connected = pyqtSignal()

    def __init__(self, session:Session):
        super().__init__()
        self.CAMS_IN_PROCESS = False

        self.session = session

        if self.session.is_capture_volume_eligible():
            logger.info("able to load Capture Volume, proceeeding with load of capture volume widget")
            self.activate_capture_volume_widget()
        else:
            logger.info("Unable to load capture volume from config; load extrinsic calibration widget")
            self.activate_extrinsic_calibration_widget()

    ###################### Stereocalibration  ######################################
    def activate_extrinsic_calibration_widget(self):
        logger.info(f"Setting session mode to {SessionMode.ExtrinsicCalibration} from within subwidget")
        self.session.set_mode(SessionMode.ExtrinsicCalibration)

        if hasattr(self, "extrinsic_calibration_widget"):
            logger.info("Activate extrinsic calibration widget")
            self.extrinsic_calibration_widget.deleteLater()
            self.extrinsic_calibration_widget = None
            # self.setCurrentWidget(self.extrinsic_calibration_widget)

        logger.info("Create new extrinsic calibration widget")
        self.extrinsic_calibration_widget = ExtrinsicCalibrationWidget(self.session)
        self.addWidget(self.extrinsic_calibration_widget)
        self.setCurrentWidget(self.extrinsic_calibration_widget)
        self.extrinsic_calibration_widget.calibration_complete.connect(self.activate_capture_volume_widget)
        self.extrinsic_calibration_widget.update_btn_eligibility()
        
        # self.session.unpause_synchronizer()
        

    def activate_capture_volume_widget(self):

        logger.info(f"Setting session mode to {SessionMode.CaptureVolumeOrigin} from within subwidget")
        self.session.set_mode(SessionMode.CaptureVolumeOrigin)

        if hasattr(self, "capture_volume_widget"):
            logger.info("Set current index to capture volume widget")
            self.capture_volume_widget.deleteLater()
            self.capture_volume_widget = None            

        logger.info("Creating Capture Volume widget")
        self.capture_volume_widget = CaptureVolumeWidget(self.session)
        logger.info("Adding capture volume widget to main Wizard")
        self.addWidget(self.capture_volume_widget)
        self.setCurrentWidget(self.capture_volume_widget)

        self.capture_volume_widget.recalibrate_btn.clicked.connect(
            self.activate_extrinsic_calibration_widget
        )
        
        # this will be managed elsewhere. StreamTools may or may not 
        # be loaded ...
        # self.session.pause_synchronizer()
            

