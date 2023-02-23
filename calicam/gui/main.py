
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

from calicam.session import Session, stage
from calicam.gui.wizard_charuco import WizardCharuco
from calicam.gui.camera_config.camera_tabs import CameraTabs
from calicam.gui.wizard_directory import WizardDirectory
from calicam import __root__, __app_dir__

class CalibrationWizard(QWidget):
    def __init__(self):
        super().__init__()

        # app = QApplication.instance()
        # screen = app.primaryScreen()
        # DISPLAY_WIDTH = screen.size().width()
        # DISPLAY_HEIGHT = screen.size().height()

        # self.setMinimumSize(int(DISPLAY_WIDTH * 0.45), int(DISPLAY_HEIGHT * 0.7))
        self.setWindowTitle("Camera Calibration Wizard")
        self.setWindowIcon(QIcon("calicam/gui/icons/fmc_logo.ico"))

        self.vbox = QVBoxLayout()
        self.setLayout(self.vbox)
        
        self.wizard_directory = WizardDirectory()

        self.vbox.addWidget(self.wizard_directory)
        self.wizard_directory.launch_wizard_btn.clicked.connect(self.move_to_charuco_wizard)
        
    
        
    def move_to_charuco_wizard(self):
        self.launch_session()
        self.wizard_charuco = WizardCharuco(self.session)
        

        # self.vbox.removeWidget(self.wizard_directory)
        self.wizard_directory.deleteLater()
        self.vbox.addWidget(self.wizard_charuco)   
         
    def launch_session(self):
        if self.wizard_directory.create_new_radio.isChecked():
            # only need to create a new session in the given directory:
            self.session_directory = self.wizard_directory.new_path.textbox.text()
            self.session = Session(self.session_directory)
        else:
            # need to copy over config from old directory to new directory before launching
            self.session_directory = self.wizard_directory.modified_path.textbox.text()
            old_config_path = self.wizard_directory.original_path.textbox.text() 
            shutil.copyfile(str(Path(old_config_path, "config.toml")), str(Path(self.session_directory, "config.toml")))
            self.session = Session(self.session_directory)
            
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
                
                self.summary.camera_summary.connected_cam_count.setText(str(len(self.session.cameras)))
                
                self.enable_disable_menu()
                self.configure_cameras.trigger()

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
        self.summary.camera_summary.connected_cam_count.setText("0")
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



if __name__ == "__main__":

    # config_path = Path(__root__, "sessions", "high_res_session")
    
    app = QApplication(sys.argv)
    window = CalibrationWizard()
    
    # open in a session already so you don't have to go through the menu each time
    # window.open_session(config_path)
    window.show()

    app.exec()