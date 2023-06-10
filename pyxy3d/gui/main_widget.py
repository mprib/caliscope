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

        self.setWindowTitle("PyXY3D")
        self.setWindowIcon(QIcon(str(Path(__root__, "pyxy3d/gui/icons/pyxy_logo.svg"))))
        self.setMinimumSize(500, 500)

        # File Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        # Open or New project (can just create a folder in the dialog in truly new)
        self.open_project_action = QAction("New/Open Project", self)
        self.open_project_action.triggered.connect(self.create_new_project_folder)
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

        self.cameras_menu = self.menu.addMenu("Cameras")
        self.connect_cameras_action = QAction("Connect Cameras", self)
        self.cameras_menu.addAction(self.connect_cameras_action)
        self.connect_cameras_action.setEnabled(False)

        self.disconnect_cameras_action = QAction("Disconnect Cameras", self)
        self.cameras_menu.addAction(self.disconnect_cameras_action)
        self.disconnect_cameras_action.setEnabled(False)

        self.connect_menu_actions()

        # Set up layout (based on splitter)
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # create log window which is fixed below main window
        self.docked_logger = QDockWidget("Log", self)
        self.docked_logger.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.docked_logger.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.log_widget = LogWidget()
        self.docked_logger.setWidget(self.log_widget)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docked_logger)



    def connect_menu_actions(self):
        self.exit_pyxy3d_action.triggered.connect(QApplication.instance().quit)
        self.disconnect_cameras_action.triggered.connect(self.disconnect_cameras)
        
    def disconnect_cameras(self):
        self.tab_widget.setCurrentWidget(self.charuco_widget)
        self.tab_widget.setTabEnabled(TabIndex.Charuco.value,True)
        self.tab_widget.setTabEnabled(TabIndex.Cameras.value,False)
        self.tab_widget.setTabEnabled(TabIndex.CaptureVolume.value,False)
        self.tab_widget.setTabEnabled(TabIndex.Recording.value,False)
        self.tab_widget.setTabEnabled(TabIndex.Processing.value,True)
        
        self.camera_widget = QWidget()
        self.calibrate_capture_volume_widget = QWidget()
        self.recording_widget = QWidget()
                
        self.session.set_mode(SessionMode.Charuco)
        self.session.disconnect_cameras() 
        self.disconnect_cameras_action.setEnabled(False)
        self.connect_cameras_action.setEnabled(True)

        

    def on_camera_connection(self):
        """toggle menu item available so that user can disconnect from cameras"""
        self.connect_cameras_action.setEnabled(False)
        self.disconnect_cameras_action.setEnabled(True)

    def on_tab_changed(self, index):
        logger.info(f"Switching main window to tab {index}")
        match index:
            case TabIndex.Charuco.value:
                logger.info(f"Activating Charuco Widget")
                self.session.set_mode(SessionMode.Charuco)
            case TabIndex.Cameras.value:
                logger.info(f"Activating Camera Setup Widget")
                self.session.set_mode(SessionMode.IntrinsicCalibration)
            case TabIndex.CaptureVolume.value:
                logger.info(f"Activating Calibrate Capture Volume Widget")

                if self.session.capture_volume_eligible():
                    self.calibrate_capture_volume_widget.activate_capture_volume_widget()
                else:
                    self.calibrate_capture_volume_widget.activate_extrinsic_calibration_widget()

            case TabIndex.Recording.value:
                logger.info(f"Activate Recording Mode")
                self.session.set_mode(SessionMode.Recording)
            case TabIndex.Processing.value:
                logger.info(f"Activate Processing Mode")
                self.session.set_mode(SessionMode.PostProcessing)
                # may have acquired new recordings
                self.processing_widget.update_recording_folders()

    def launch_session(self, path_to_folder: str):
        session_path = Path(path_to_folder)
        self.config = Configurator(session_path)
        logger.info(f"Launching session with config file stored in {session_path}")
        self.session = Session(self.config)
        self.connect_cameras_action.triggered.connect(self.session.connect_to_cameras)
        self.session.cameras_connected_signal.connect(self.update_tabs_eligibility)
        self.session.cameras_connected_signal.connect(self.on_camera_connection)

        # can always load charuco
        self.charuco_widget = CharucoWidget(self.session)

        # Only active tab has a real widget, everything else is a placeholder that
        # is built as needed from scratch when that tab is selected
        self.camera_widget = QWidget()
        self.calibrate_capture_volume_widget = QWidget()
        self.recording_widget = QWidget()
        self.processing_widget = QWidget()
        

        self.tab_widget.addTab(self.charuco_widget, "Charuco")
        self.tab_widget.addTab(self.camera_widget, "Cameras")
        self.tab_widget.addTab(self.calibrate_capture_volume_widget, "CaptureVolume")
        self.tab_widget.addTab(self.recording_widget, "Recording")
        self.tab_widget.addTab(self.processing_widget, "Processing")

        # when tabs change, make sure session mode adjusts
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Default to having ability to search out stream tools
        self.connect_cameras_action.setEnabled(True)
        self.update_tabs_eligibility()
        # might be able to do
        old_index = self.tab_widget.currentIndex()

        self.tab_widget.setCurrentIndex(old_index)
        self.connect_session_signals()

    def update_tabs_eligibility(self):
        """
        current state of the session defines what functionality the GUI should be
        capable of moving into... this method makes sure that the user cannot 
        initiate a mode/GUI widget that cannot be created by the session.
        """

        # can always modify charuco and access postprocessing
        self.tab_widget.setTabEnabled(TabIndex.Charuco.value, True)
        # self.tab_widget.setTabEnabled(TabIndex.Processing.value, True)
        
        # for now, disable just because that widget needs fixing...
        self.tab_widget.setTabEnabled(TabIndex.Processing.value, False)

        # if you are connected to comeras, can configure them
        # recording widget may not be able to actually record if not calibrated
        if self.session.camera_setup_eligible():
            self.tab_widget.setTabEnabled(TabIndex.Cameras.value, True)
            self.tab_widget.setTabEnabled(TabIndex.Recording.value, True)
        else:
            self.tab_widget.setTabEnabled(TabIndex.Cameras.value, False)
            self.tab_widget.setTabEnabled(TabIndex.Recording.value, False)
        
        if self.session.fully_calibrated_intrinsics():
            self.tab_widget.setTabEnabled(TabIndex.CaptureVolume.value, True)
        else:
            self.tab_widget.setTabEnabled(TabIndex.CaptureVolume.value, False)



    def connect_session_signals(self):
        """
        After launching a session, connect signals and slots.
        Much of these will be from the GUI to the session and vice-versa
        """
        self.session.unlock_postprocessing.connect(self.load_post_processing_widget)
        self.session.stream_tools_loaded_signal.connect(self.update_tabs_eligibility)

    def load_recording_widget(self):
        # recording_index = self.tab_widget.indexOf(self.recording_widget)
        self.tab_widget.removeTab(TabIndex.Recording.value)
        self.recording_widget.deleteLater()
        new_recording_widget = RecordingWidget(self.session)
        self.tab_widget.insertTab(
            TabIndex.Recording.value, new_recording_widget, TabIndex.Recording.name
        )
        self.recording_widget = new_recording_widget

    def load_post_processing_widget(self):
        self.tab_widget.removeTab(TabIndex.Processing.value)
        self.processing_widget.deleteLater()
        new_processing_widget = PostProcessingWidget(self.session)
        self.tab_widget.insertTab(
            TabIndex.Processing.value, new_processing_widget, TabIndex.Processing.name
        )
        self.processing_widget = new_processing_widget

    def load_capture_volume_widget(self):
        self.tab_widget.removeTab(TabIndex.CaptureVolume.value)
        self.calibrate_capture_volume_widget.deleteLater()
        new_capture_volume_widget = CalibrateCaptureVolumeWidget(self.session)
        self.tab_widget.insertTab(
            TabIndex.Cameras.value,
            new_capture_volume_widget,
            TabIndex.CaptureVolume.name,
        )
        self.camera_widget = new_capture_volume_widget

    def load_camera_widget(self):
        self.tab_widget.removeTab(TabIndex.Cameras.value)
        self.camera_widget.deleteLater()
        new_camera_widget = IntrinsicCalibrationWidget(self.session)
        self.tab_widget.insertTab(
            TabIndex.Cameras.value, new_camera_widget, TabIndex.Cameras.name
        )
        self.camera_widget = new_camera_widget

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
