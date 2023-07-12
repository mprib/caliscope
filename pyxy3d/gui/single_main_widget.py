import pyxy3d.logger
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from PyQt6.QtWidgets import QMainWindow, QStackedLayout, QFileDialog

logger = pyxy3d.logger.get(__name__)
from pathlib import Path
from threading import Thread
import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStackedLayout,
    QWidget,
    QDockWidget,
    QVBoxLayout,
    QMenu,
    QMenuBar,
    QTabWidget,
)
import toml
from enum import Enum
from PyQt6.QtGui import QIcon, QAction, QKeySequence, QShortcut
from PyQt6.QtCore import Qt
from pyxy3d import __root__, __settings_path__, __user_dir__
from pyxy3d.session.session import Session, SessionMode
from pyxy3d.gui.log_widget import LogWidget
from pyxy3d.configurator import Configurator
from pyxy3d.gui.charuco_widget import CharucoWidget
from pyxy3d.gui.camera_config.intrinsic_calibration_widget import (
    IntrinsicCalibrationWidget,
)
from pyxy3d.gui.calibrate_capture_volume_widget import CalibrateCaptureVolumeWidget
from pyxy3d.gui.recording_widget import RecordingWidget
from pyxy3d.gui.post_processing_widget import PostProcessingWidget
from pyxy3d.gui.extrinsic_calibration_widget import ExtrinsicCalibrationWidget
from pyxy3d.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget

class TabIndex(Enum):
    Charuco = 0
    Cameras = 1
    CaptureVolume = 2
    Recording = 3
    Processing = 4


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.app_settings = toml.load(__settings_path__)

        self.setWindowTitle("Pyxy3D")
        self.setWindowIcon(QIcon(str(Path(__root__, "pyxy3d/gui/icons/pyxy_logo.svg"))))
        self.setMinimumSize(500, 500)

        # File Menu
        self.menu = self.menuBar()
        
        # CREATE FILE MENU
        self.file_menu = self.menu.addMenu("&File")
        self.open_project_action = QAction("New/Open Project", self)
        self.file_menu.addAction(self.open_project_action)

        # Open Recent
        self.open_recent_project_submenu = QMenu("Recent Projects...", self)
        # Populate the submenu with recent project paths;
        # reverse so that last one appended is at the top of the list
        for project_path in reversed(self.app_settings["recent_projects"]):
            self.add_to_recent_project(project_path)

        self.file_menu.addMenu(self.open_recent_project_submenu)

        self.exit_pyxy3d_action = QAction("Exit", self)
        self.file_menu.addAction(self.exit_pyxy3d_action)


        # CREATE CAMERA MENU
        self.cameras_menu = self.menu.addMenu("&Cameras")
        self.connect_cameras_action = QAction("Co&nnect Cameras", self)
        self.cameras_menu.addAction(self.connect_cameras_action)
        self.connect_cameras_action.setEnabled(False)

        self.disconnect_cameras_action = QAction("&Disconnect Cameras", self)
        self.cameras_menu.addAction(self.disconnect_cameras_action)
        self.disconnect_cameras_action.setEnabled(False)

        # CREATE MODE MENU
        self.mode_menu = self.menu.addMenu("&Mode")
        self.charuco_mode_select = QAction(SessionMode.Charuco.value)
        self.intrinsic_mode_select = QAction(SessionMode.IntrinsicCalibration.value)
        self.extrinsic_mode_select = QAction(SessionMode.ExtrinsicCalibration.value)
        self.capture_volume_mode_select = QAction(SessionMode.CaptureVolumeOrigin.value)
        self.recording_mode_select = QAction(SessionMode.Recording.value)
        self.processing_mode_select = QAction(SessionMode.PostProcessing.value)
        self.mode_menu.addAction(self.charuco_mode_select)
        self.mode_menu.addAction(self.intrinsic_mode_select)
        self.mode_menu.addAction(self.extrinsic_mode_select)
        self.mode_menu.addAction(self.capture_volume_mode_select)
        self.mode_menu.addAction(self.recording_mode_select)
        self.mode_menu.addAction(self.processing_mode_select)

        for action in self.mode_menu.actions():
            action.setEnabled(False)

        self.connect_menu_actions()
        self.blank_widget = QWidget()
        self.setCentralWidget(self.blank_widget)

        # create log window which is fixed below main window
        self.docked_logger = QDockWidget("Log", self)
        self.docked_logger.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.docked_logger.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.log_widget = LogWidget()
        self.docked_logger.setWidget(self.log_widget)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docked_logger)

    def connect_menu_actions(self):
        self.open_project_action.triggered.connect(self.create_new_project_folder)
        self.connect_cameras_action.triggered.connect(self.load_stream_tools)
        self.exit_pyxy3d_action.triggered.connect(QApplication.instance().quit)
        self.disconnect_cameras_action.triggered.connect(self.disconnect_cameras)

        for action in self.mode_menu.actions():
            action.triggered.connect(self.mode_change_action)

    def mode_change_action(self):
        action = self.sender()
        SessionModeLookup = {mode.value: mode for mode in SessionMode}
        mode = SessionModeLookup[action.text()]
        self.session.set_mode(mode)
            
        logger.info(f"Attempting to set session mode to {mode.value}")
         
    def update_central_widget_mode(self):
        # Delete the current central widget
        old_widget = self.centralWidget()
        self.setCentralWidget(None)
        old_widget.deleteLater()

        if type(old_widget) == RecordingWidget:
            old_widget.thumbnail_emitter.keep_collecting.clear()
        
        # Create the new central widget based on the mode
        match self.session.mode:
            case SessionMode.Charuco:
                new_widget = CharucoWidget(self.session)
            case SessionMode.IntrinsicCalibration:
                new_widget = IntrinsicCalibrationWidget(self.session)
            case SessionMode.ExtrinsicCalibration:
                new_widget = ExtrinsicCalibrationWidget(self.session)    
            case SessionMode.CaptureVolumeOrigin:
                new_widget = CaptureVolumeWidget(self.session)
            case SessionMode.Recording:
                new_widget = RecordingWidget(self.session)
            case SessionMode.PostProcessing:
                new_widget = PostProcessingWidget(self.session)
            
        self.setCentralWidget(new_widget)        

    def disconnect_cameras(self):
                
        self.session.set_mode(SessionMode.Charuco)
        self.session.disconnect_cameras() 
        self.disconnect_cameras_action.setEnabled(False)
        self.connect_cameras_action.setEnabled(True)

        self.intrinsic_mode_select.setEnabled(False)
        self.extrinsic_mode_select.setEnabled(False)

    def pause_all_frame_reading(self):
        logger.info("Pausing all frame reading at load of stream tools; should be on charuco tab right now")
        self.session.pause_all_monocalibrators()
        self.session.pause_synchronizer()  

    def load_stream_tools(self):
        self.connect_cameras_action.setEnabled(False)
        self.disconnect_cameras_action.setEnabled(True)
        self.session.qt_signaler.stream_tools_loaded_signal.connect(self.pause_all_frame_reading)
        self.thread = Thread(
            target=self.session.load_stream_tools, args=(), daemon=True
        )
        self.thread.start()
            

    def launch_session(self, path_to_folder: str):
        session_path = Path(path_to_folder)
        self.config = Configurator(session_path)
        logger.info(f"Launching session with config file stored in {session_path}")
        self.session = Session(self.config)

        # can always load charuco
        self.charuco_mode_select.setEnabled(True) 
        self.charuco_widget = CharucoWidget(self.session)
        self.setCentralWidget(self.charuco_widget)

        # now connecting to cameras is an option
        self.connect_cameras_action.setEnabled(True) 
        
        # but must exit and start over to launch a new session for now
        self.open_project_action.setEnabled(False) 
        self.open_recent_project_submenu.setEnabled(False)
        self.connect_session_signals()


    def connect_session_signals(self):
        """
        After launching a session, connect signals and slots.
        Much of these will be from the GUI to the session and vice-versa
        """
        self.session.qt_signaler.unlock_postprocessing.connect(self.enable_post_processing)
        self.session.qt_signaler.stream_tools_loaded_signal.connect(self.enable_camera_tools)
        self.session.qt_signaler.mode_change_success.connect(self.update_central_widget_mode)

    def enable_post_processing(self):
        self.processing_mode_select.setEnabled(True)
    
    def enable_camera_tools(self):
        self.intrinsic_mode_select.setEnabled(True)
        self.extrinsic_mode_select.setEnabled(True)
        self.recording_mode_select.setEnabled(True)
        
    def add_to_recent_project(self, project_path: str):
        recent_project_action = QAction(project_path, self)
        recent_project_action.triggered.connect(self.open_recent_project)
        self.open_recent_project_submenu.addAction(recent_project_action)

    def open_recent_project(self):
        action = self.sender()
        project_path = action.text()
        logger.info(f"Opening recent session stored at {project_path}")
        self.launch_session(project_path)

    def create_new_project_folder(self):
        default_folder = Path(self.app_settings["last_project_parent"])
        dialog = QFileDialog()
        path_to_folder = dialog.getExistingDirectory(
            parent=None,
            caption="Open Previous or Create New Project Directory",
            directory=str(default_folder),
            options=QFileDialog.Option.ShowDirsOnly,
        )

        if path_to_folder:
            logger.info(("Creating new project in :", path_to_folder))
            self.add_project_to_recent(path_to_folder)
            self.launch_session(path_to_folder)

    def add_project_to_recent(self, folder_path):
        if str(folder_path) in self.app_settings["recent_projects"]:
            pass
        else:
            self.app_settings["recent_projects"].append(str(folder_path))
            self.app_settings["last_project_parent"] = str(Path(folder_path).parent)
            self.update_app_settings()
            self.add_to_recent_project(folder_path)

    def update_app_settings(self):
        with open(__settings_path__, "w") as f:
            toml.dump(self.app_settings, f)

def launch_main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    app.exec()


if __name__ == "__main__":
    launch_main()
