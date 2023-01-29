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

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QStackedWidget,
    QMainWindow,
)

from calicam.session import Session, stage
from calicam.gui.left_sidebar.session_summary import SessionSummary
from calicam.gui.charuco_builder import CharucoBuilder
from calicam.gui.camera_config.camera_tabs import CameraTabs
from calicam.gui.stereo_calibration.stereo_cal_dialog import StereoCalDialog

class MainWindow(QMainWindow):
    def __init__(self, session=None):
        super().__init__()
        self.repo = Path(__file__).parent.parent.parent
        if session is not None:
            self.session = session

        app = QApplication.instance()
        screen = app.primaryScreen()
        DISPLAY_WIDTH = screen.size().width()
        DISPLAY_HEIGHT = screen.size().height()

        self.setMinimumSize(int(DISPLAY_WIDTH * 0.45), int(DISPLAY_HEIGHT * 0.7))
        self.setWindowTitle("FreeMocap Camera Calibration")
        self.setWindowIcon(QIcon("src/gui/icons/fmc_logo.ico"))

        self.menu = self.menuBar()
        self.central_stack = QStackedWidget()
        self.setCentralWidget(self.central_stack)

        self.CAMS_IN_PROCESS = False
        self.stage = stage.NO_CAMERAS
    
        self.build_file_menu()
        self.build_view_menu()
        self.build_actions_menu()
        
        self.enable_disable_menu()
         
    def build_file_menu(self):
        
        file = self.menu.addMenu("&File")
        file_new_session = QAction("Create &New Session", self)
        file_new_session.triggered.connect(self.get_session)
        file.addAction(file_new_session)
        
        file_saved_session = QAction("&Open Saved Session", self)
        file_saved_session.triggered.connect(self.get_session)
        file.addAction(file_saved_session)

    def get_session(self):
        
        logging.info("Prompting for session path...")
        sessions_directory = str(Path(self.repo, "sessions"))
        session_path = QFileDialog.getExistingDirectory(
            self, "Select Session Folder", sessions_directory
        )

        self.open_session(session_path)

    def build_view_menu(self): 
        view = self.menu.addMenu("&View")

        build_charuco = QAction("&Build Charuco", self)
        view.addAction(build_charuco)
        build_charuco.triggered.connect(self.activate_charuco_builder)

        self.configure_cameras = QAction("Configure &Cameras", self)
        view.addAction(self.configure_cameras)
        self.configure_cameras.triggered.connect(self.launch_cam_config_dialog)
        
        self.stereocalibrate = QAction("&Stereocalibrate", self)
        view.addAction(self.stereocalibrate)
        self.stereocalibrate.triggered.connect(self.launch_stereocal_dialog)
    
    def launch_stereocal_dialog(self):
        logging.info("Launching stereocalibration dialog")
        # if hasattr(self, "stereo_cal_dialog"):
            # self.central_stack.setCurrentWidget(self.stereo_cal_dialog)
        # else:
            # self.session.load_synchronizer()
        self.session.remove_monocalibrators()
        if hasattr(self,"camera_tabs"):
            logging.info("Removing camera tabs")
            del self.camera_tabs

        self.session.load_stereo_tools()
        self.stereo_cal_dialog = StereoCalDialog(self.session)
        self.central_stack.addWidget(self.stereo_cal_dialog)
        self.central_stack.setCurrentWidget(self.stereo_cal_dialog)

    def build_actions_menu(self):
        actions = self.menu.addMenu("&Actions")
        # self.menu.addMenu(actions)

        self.connect_cameras_action = QAction("Connect to &Saved Cameras", self)
        actions.addAction(self.connect_cameras_action)
        self.connect_cameras_action.triggered.connect(self.connect_to_cameras)
        
        self.find_additional_action = QAction("&Find Cameras", self)
        actions.addAction(self.find_additional_action)
        self.find_additional_action.triggered.connect(self.find_cameras)
        
        self.disconnect_cam_action = QAction("&Disconnect Cameras", self)
        actions.addAction(self.disconnect_cam_action)
        self.disconnect_cam_action.triggered.connect(self.disconnect_cameras)

        self.record_action = QAction("&Record")
        actions.addAction(self.record_action)
        self.record_action.triggered.connect(self.start_stop_recording)
        
    
    def start_stop_recording(self):
        if not hasattr(self.session, "video_recorder"):
            self.session.load_video_recorder()
        if not self.session.video_recorder.recording:
            self.session.video_recorder.start_recording(Path(self.session.path, "recording"))
            self.record_action.setText("Stop &Recording")
        else:
            self.session.video_recorder.stop_recording()
            self.record_action.setText("&Record")
    
    def open_session(self, session_path):
        """The primary action of choosing File--Open or New session"""
        try:
            # self.summary.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose,True)
            self.summary.close()
        except(AttributeError):
            pass
        
        logging.info(f"Opening session located at {session_path}")
        self.session = Session(session_path)
        self.summary = SessionSummary(self.session)
        self.enable_disable_menu()
        
        self.dock = QDockWidget("Session Summary", self)
        self.dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.dock.setWidget(self.summary)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)

    def enable_disable_menu(self):
        if not hasattr(self, "session"):
            self.configure_cameras.setEnabled(False)
            self.disconnect_cam_action.setEnabled(False) 
            self.connect_cameras_action.setEnabled(False)
            self.find_additional_action.setEnabled(False)
            self.stereocalibrate.setEnabled(False)
            return
        
        self.summary.stage_label.setText(f"Stage: {self.session.get_stage()}")
        if self.session.get_stage() == stage.NO_CAMERAS:
            self.configure_cameras.setEnabled(False)
            self.disconnect_cam_action.setEnabled(False) 
            self.connect_cameras_action.setEnabled(True)
            self.find_additional_action.setEnabled(True)
            self.stereocalibrate.setEnabled(False)
        else:
            self.configure_cameras.setEnabled(True)
            self.disconnect_cam_action.setEnabled(True) 
            self.connect_cameras_action.setEnabled(False)
            self.find_additional_action.setEnabled(False)
        
        if self.session.get_stage().value >= stage.MONOCALIBRATED_CAMERAS.value:
            self.stereocalibrate.setEnabled(True)        
        
    def launch_cam_config_dialog(self):
        logging.info("Launching camera configuration dialog tabs")

        while self.CAMS_IN_PROCESS:
            logging.warning("Cams in process and waiting")
            time.sleep(.3)
        
        # self.camera_tabs = None
        if not hasattr(self,"camera_tabs"):
            self.camera_tabs = CameraTabs(self.session)
            
            def on_save_cam_click():
                self.summary.camera_summary.camera_table.update_data()
                self.enable_disable_menu() 

            for tab_index in range(self.camera_tabs.count()):
                self.camera_tabs.widget(tab_index).save_cal_btn.clicked.connect(on_save_cam_click)
            
            self.central_stack.addWidget(self.camera_tabs)
            self.central_stack.setCurrentWidget(self.camera_tabs) 
        else:
            logging.info("Camera tabs already exist....not deleting")
            self.central_stack.setCurrentWidget(self.camera_tabs)
        
    
    def connect_to_cameras(self):

        if len(self.session.cameras) > 0:
            logging.info("Cameras already connected")
            pass
        else:

            def connect_to_cams_worker():
                self.CAMS_IN_PROCESS = True
                logging.info("Initiating camera connect worker")
                self.session.load_cameras()
                logging.info("Camera connect worker about to load stream tools")
                self.session.load_streams()
                logging.info("Camera connect worker about to adjust resolutions")
                self.session.adjust_resolutions()
                logging.info("Camera connect worker about to load monocalibrators")
                self.session.load_monocalibrators()
                self.CAMS_IN_PROCESS = False
                
                self.summary.camera_summary.connected_cam_count.setText(str(len(self.session.cameras)))
                
                self.enable_disable_menu()
                self.configure_cameras.trigger()

        if self.CAMS_IN_PROCESS:
            logging.info("Already attempting to connect to cameras...")
        else:
            self.connect_cams = Thread(target = connect_to_cams_worker, args=[], daemon=True)
            self.connect_cams.start()
            
    def disconnect_cameras(self):
        logging.info("Attempting to disconnect cameras")

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
            logging.info("Loading streams")
            self.session.load_streams()
            logging.info("Loading monocalibrators")
            self.session.load_monocalibrators()
            logging.info("Updating Camera Table")
            self.summary.camera_summary.camera_table.update_data()

            self.CAMS_IN_PROCESS = False
            self.summary.camera_summary.connected_cam_count.setText(str(len(self.session.cameras)))
            self.enable_disable_menu()
            self.configure_cameras.trigger()
            
        if self.CAMS_IN_PROCESS:
            logging.info("Cameras already connected or in process.")        
        else:
            logging.info("Searching for additional cameras...This may take a moment.")
            self.find = Thread(target=find_cam_worker, args=(), daemon=True)
            self.find.start()

    def create_charuco_builder(self):

        self.charuco_builder = CharucoBuilder(self.session)
        self.central_stack.addWidget(self.charuco_builder)
        self.central_stack.setCurrentWidget(self.charuco_builder)

        def update_summary():
            self.summary.charuco_summary.update_charuco_summary()
        
        self.charuco_builder.save_btn.clicked.connect(update_summary)
        
        self.CHARUCO_BUILDER_MADE = True


    def activate_charuco_builder(self):
        if hasattr(self, "charuco_builder"):
            self.central_stack.setCurrentWidget(self.charuco_builder)
        else:
            self.create_charuco_builder()

def launch_main_window():
    repo = Path(__file__).parent.parent.parent
    config_path = Path(repo, "sessions", "high_res_session")
    
    app = QApplication(sys.argv)
    window = MainWindow()
    
    # open in a session already so you don't have to go through the menu each time
    # window.open_session(config_path)
    window.show()
    window.connect_cameras_action.trigger()

    app.exec()
if __name__ == "__main__":
    launch_main_window()