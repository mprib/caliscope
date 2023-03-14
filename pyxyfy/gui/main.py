

import shutil
import sys
from pathlib import Path
from threading import Thread

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QStackedWidget,
)

import pyxyfy.logger
from pyxyfy import __root__
from pyxyfy.gui.camera_config.camera_tabs import CameraWizard
from pyxyfy.gui.omniframe.omni_frame_widget import OmniFrameWidget
from pyxyfy.gui.qt_logger import QtLogger
from pyxyfy.gui.wizard_charuco import WizardCharuco
from pyxyfy.gui.wizard_directory import WizardDirectory
from pyxyfy.session import Session

logger = pyxyfy.logger.get(__name__)

class CalibrationWizard(QStackedWidget):
    cameras_connected = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.CAMS_IN_PROCESS = False

        self.setWindowTitle("Camera Calibration Wizard")
        self.setWindowIcon(QIcon(str(Path(__root__, "pyxyfy/gui/icons/orb.svg"))))
        self.wizard_directory = WizardDirectory()
        self.addWidget(self.wizard_directory) # index:1
        self.setCurrentIndex(0)
        self.connect_widgets()
        
        
    def connect_widgets(self):
        self.wizard_directory.launch_wizard_btn.clicked.connect(self.next_to_charuco_wizard)
        self.cameras_connected.connect(self.on_cameras_connect) 
        
    def next_to_omniframe(self):
        self.omniframe = OmniFrameWidget(self.session)
        self.addWidget(self.omniframe)
        self.setCurrentIndex(3)
        self.omniframe.navigation_bar.back_btn.clicked.connect(self.move_back_to_camera_config_wizard)

    def on_cameras_connect(self):
        # load cameras wizard once the cameras are actually connected
        self.camera_wizard = CameraWizard(self.session)
        self.addWidget(self.camera_wizard)
        self.setCurrentIndex(2)
        self.camera_wizard.navigation_bar.back_btn.clicked.connect(self.back_to_charuco_wizard)
        self.camera_wizard.navigation_bar.next_btn.clicked.connect(self.next_to_omniframe)
     
    def back_to_charuco_wizard(self):
        self.setCurrentIndex(1)

    def next_to_charuco_wizard(self):
        if hasattr(self, "wizard_charuco"):
            self.setCurrentIndex(1)
        else:
            logger.info("Launching session")
            self.launch_session()
            logger.info("Creating charuco wizard session")
            self.wizard_charuco = WizardCharuco(self.session)
            self.wizard_charuco.navigation_bar.next_wizard_step_btn.clicked.connect(self.move_next_to_camera_config_wizard)
            logger.info("Adding charuco wizard")
            self.addWidget(self.wizard_charuco)
            logger.info("Setting index to 2 to activate widget")
            self.setCurrentIndex(1)

    def move_next_to_camera_config_wizard(self):
        if hasattr(self, "camera_wizard"):
            logger.info("Camera wizard already exists; changing stack current index")
            self.setCurrentIndex(2)
            logger.info("updating charuco in case necessary")
            for port, stream in self.session.streams.items():
                stream.update_charuco(self.session.charuco)
        else:
            logger.info("Initiating Camera Connection")
            self.initiate_camera_connection()
            self.qt_logger = QtLogger("Connecting to Cameras")
            self.qt_logger.show()
   
    def move_back_to_camera_config_wizard(self):
        # from omniframe to camera config
        self.setCurrentIndex(2)
        del self.session.synchronizer
                     
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
                shutil.copyfile(str(Path(old_config_path, "config.toml")), str(Path(self.session_directory, "config.toml")))

            self.session = Session(self.session_directory)


    def initiate_camera_connection(self):

        if len(self.session.cameras) > 0:
            logger.info("Cameras already connected")
        else:

            def connect_to_cams_worker():
                self.CAMS_IN_PROCESS = True
                logger.info("Initiating camera connect worker")
                
                # find out if you are loading cameras or finding cameras
                if self.session.get_configured_camera_count()>0:
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
            self.connect_cams = Thread(target = connect_to_cams_worker, args=[], daemon=True)
            self.connect_cams.start()
            
    def disconnect_cameras(self):
        logger.info("Attempting to disconnect cameras")

        if hasattr(self, "camera_tabs"):
            self.central_stack.removeWidget(self.camera_tabs) 
            del self.camera_tabs 
        if hasattr(self, "stereo_cal_dialog"):
            self.central_stack.removeWidget(self.stereo_cal_dialog)
            del self.stereo_cal_dialog
        self.session.disconnect_cameras()
        # self.summary.camera_summary.connected_cam_count.setText("0")
        self.enable_disable_menu()

    def find_cameras(self):
        def find_cam_worker():
            self.CAMS_IN_PROCESS = True
            self.session.find_cameras()
            logger.info("Loading streams")
            self.session.load_streams()
            logger.info("Loading monocalibrators")
            self.session.load_monocalibrators()
            logger.info("Updating Camera Table")
            self.summary.camera_summary.camera_table.update_data()

            self.CAMS_IN_PROCESS = False
            self.summary.camera_summary.connected_cam_count.setText(str(len(self.session.cameras)))
            self.enable_disable_menu()
            self.configure_cameras.trigger()
            
        if self.CAMS_IN_PROCESS:
            logger.info("Cameras already connected or in process.")        
        else:
            logger.info("Searching for additional cameras...This may take a moment.")
            self.find = Thread(target=find_cam_worker, args=(), daemon=True)
            self.find.start()

def launch_pyxyfy():

    app = QApplication(sys.argv)
    window = CalibrationWizard()
    window.show()

    app.exec()

if __name__ == "__main__":

    launch_pyxyfy()