
import calicam.logger
logger = calicam.logger.get(__name__)

import sys
import shutil
import time
from pathlib import Path
from threading import Thread
from PyQt6.QtCore import Qt
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

from calicam.session import Session, Stage
from calicam.gui.wizard_charuco import WizardCharuco
from calicam.gui.camera_config.camera_tabs import CameraWizard
from calicam.gui.wizard_directory import WizardDirectory
from calicam import __root__, __app_dir__
from calicam.session import Stage

class CalibrationWizard(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.CAMS_IN_PROCESS = False

        self.setWindowTitle("Camera Calibration Wizard")
        self.setWindowIcon(QIcon("calicam/gui/icons/fmc_logo.ico"))
        # self.vbox = QVBoxLayout()
        # self.setLayout(self.vbox)
        # land on the directory selector widget        
        self.wizard_directory = WizardDirectory()
        self.addWidget(self.wizard_directory)
        self.setCurrentIndex(1)
        # self.vbox.addWidget(self.wizard_directory)
        # link to charuco widget for next step
        self.wizard_directory.launch_wizard_btn.clicked.connect(self.next_to_charuco_wizard)
        
        
    def back_to_charuco_wizard(self):
        self.setCurrentIndex(2)
        # self.vbox.removeWidget(self.wizard_directory)
        self.wizard_cameras.hide()
        self.wizard_charuco.show()
        # self.vbox.addWidget(self.wizard_charuco)   

    def next_to_charuco_wizard(self):
        # directory will be set now
        logger.info("Launching session")
        self.launch_session()
        logger.info("Adding charuco wizard")
        self.addWidget(self.wizard_charuco)   
        self.setCurrentIndex(2)

    def move_next_to_camera_config_wizard(self):
        if self.session.get_stage() == Stage.NO_CAMERAS:
            self.connect_to_cameras()
            self.addWidget(self.wizard_cameras)
            self.setCurrentIndex(3)
        else:
            self.setCurrentIndex(3)
         
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

        self.link_widgets_to_session()
            
    def link_widgets_to_session(self):
        # once the session directory is set, all of the widgets can be linked upjfkdljkl:
        logger.info("creating charuco wizard")
        self.wizard_charuco = WizardCharuco(self.session)
        logger.info("Adding charuco wizard to qstackedwidget")
        self.addWidget(self.wizard_charuco)
        self.wizard_charuco.navigation_bar.next_wizard_step_btn.clicked.connect(self.move_next_to_camera_config_wizard)
        logger.info("creating camera wizard")
        self.wizard_cameras = CameraWizard(self.session)
        logger.info("adding camera wizard to qstackedwidget")
        self.addWidget(self.wizard_cameras)

        self.wizard_cameras.navigation_bar.back_btn.clicked.connect(self.back_to_charuco_wizard)
        self.wizard_cameras.navigation_bar.next_btn.setEnabled(False)
    

    def connect_to_cameras(self):

        if len(self.session.cameras) > 0:
            logger.info("Cameras already connected")
            pass
        else:

            def connect_to_cams_worker():
                self.CAMS_IN_PROCESS = True
                logger.info("Initiating camera connect worker")
                self.session.load_cameras()
                logger.info("Camera connect worker about to load stream tools")
                self.session.load_streams()
                logger.info("Camera connect worker about to adjust resolutions")
                self.session.adjust_resolutions()
                logger.info("Camera connect worker about to load monocalibrators")
                self.session.load_monocalibrators()
                self.CAMS_IN_PROCESS = False

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

def launch_calicam():

    test_session = Path(__root__, "sessions", "laptop")

    app = QApplication(sys.argv)
    window = CalibrationWizard()
    window.wizard_directory.from_previous_radio.click()
    window.wizard_directory.from_previous_radio.setChecked(True)
    window.wizard_directory.original_path.textbox.setText(str(test_session))
    window.wizard_directory.modified_path.textbox.setText(str(test_session))
    window.show()

    app.exec()

if __name__ == "__main__":

    # config_path = Path(__root__, "sessions", "high_res_session")
    
    
    # open in a session already so you don't have to go through the menu each time
    # window.open_session(config_path)
    launch_calicam()