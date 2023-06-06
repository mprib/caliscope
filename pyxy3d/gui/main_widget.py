import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from PyQt6.QtWidgets import QMainWindow, QStackedLayout, QFileDialog

logger = pyxy3d.logger.get(__name__)
from pathlib import Path

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
from PyQt6.QtGui import QIcon, QAction, QKeySequence, QShortcut
from PyQt6.QtCore import Qt
from pyxy3d import __root__, __settings_path__, __user_dir__
from pyxy3d.session.session import Session, SessionMode
from pyxy3d.gui.log_widget import LogWidget
from pyxy3d.configurator import Configurator
from pyxy3d.gui.calibration_widget import CalibrationWidget
from pyxy3d.gui.charuco_widget import CharucoWidget
from pyxy3d.gui.camera_config.intrinsic_calibration_widget import IntrinsicCalibrationWidget
from pyxy3d.gui.extrinsic_calibration_widget import ExtrinsicCalibrationWidget
from pyxy3d.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget
from pyxy3d.gui.recording_widget import RecordingWidget
from pyxy3d.gui.post_processing_widget import PostProcessingWidget



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.app_settings = toml.load(__settings_path__)

        self.setWindowTitle("PyXY3D")
        self.setWindowIcon(QIcon(str(Path(__root__, "pyxy3d/gui/icons/pyxy_logo.svg"))))
        self.setMinimumSize(500, 500)

        # File Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("&File")

        # Open or New project (can just create a folder in the dialog in truly new)
        self.open_project_action = QAction("&New/Open Project", self)
        self.open_project_action.triggered.connect(self.create_new_project_folder)
        self.file_menu.addAction(self.open_project_action)

        # Open Recent
        self.open_recent_project_submenu = QMenu("&Recent Projects...", self)
        # Populate the submenu with recent project paths;
        # reverse so that last one appended is at the top of the list
        for project_path in reversed(self.app_settings["recent_projects"]):
            self.add_to_recent_project(project_path)

        self.file_menu.addMenu(self.open_recent_project_submenu)

        self.close_session_action = QAction("&Close Session", self)
        self.close_session_action.triggered.connect(self.close_current_session)
        self.file_menu.addAction(self.close_session_action)

        self.cameras_menu = self.menu.addMenu("Ca&meras")
        self.disconnect_cameras_action = QAction("&Disconnect Cameras", self)
        self.connect_cameras_action = QAction("Co&nnect Cameras", self)
        self.cameras_menu.addAction(self.disconnect_cameras_action)
        self.cameras_menu.addAction(self.connect_cameras_action)

        # Set up layout (based on splitter)
        # central_widget = QWidget(self)

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # create log window which is fixed below main window
        self.docked_logger = QDockWidget("Log", self)
        self.docked_logger.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.docked_logger.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.log_widget = LogWidget()
        self.docked_logger.setWidget(self.log_widget)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docked_logger)



    ################## FRAME READING and TRACKING CONTROL with TAB SWITCH ######################################


    def on_tab_changed(self, index):
        logger.info(f"Switching main window to tab {index}")
        match index:
            case 0:
                logger.info(f"Activate Calibration Mode")
                if hasattr(self.calibration_widget, "currentWidget"):
                    match self.calibration_widget.currentWidget():
                        case self.calibration_widget.intrinsic_calibration_widget:
                            active_camera = (
                                self.calibration_widget.intrinsic_calibration_widget.camera_tabs.currentWidget().port
                            )
                            logger.info(
                                f"Activating intrinsic calibration tab: camera config widget with Camera {active_camera} active"
                            )
                            self.session.set_mode(SessionMode.IntrinsicCalibration)
                        case self.calibration_widget.extrinsic_calibration_widget:
                            logger.info(
                                "Activating extrinsic calibration tab: stereoframe widget"
                            )
                            self.session.set_mode(SessionMode.ExtrinsicCalibration)

            case 1:
                logger.info(f"Activate Recording Mode")
                self.session.set_mode(SessionMode.Recording)
            case 2:
                logger.info(f"Activate Processing Mode")
                self.session.set_mode(SessionMode.PostProcessing)
                # may have acquired new recordings
                self.processing_widget.update_recording_folders()

    def close_current_session(self):
        pass

    def launch_session(self, path_to_folder: str):
        session_path = Path(path_to_folder)
        self.config = Configurator(session_path)
        logger.info(f"Launching session with config file stored in {session_path}")
        self.session = Session(self.config)
        logger.info("Setting calibration Widget")
        self.session.stream_tools_loaded_signal.connect(self.load_recording_widget)

        # self.calibration_widget = CalibrationWidget(self.session)
        # can always load charuco
        
        self.charuco_widget = CharucoWidget(self.session)

        # launches without cameras connected, so just throw in a placeholder
        self.camera_widget = QWidget()

        self.recording_widget = QWidget()
        if self.session.post_processing_eligible():
            self.processing_widget = PostProcessingWidget(self.session)
        else:
            self.processing_widget = QWidget()
        
        if self.session.capture_volume_eligible():
            self.capture_volume_widget = CaptureVolumeWidget(self.session)
        else:
            self.capture_volume_widget = QWidget()

        self.tab_widget.addTab(self.charuco_widget, "Charuco")
        self.tab_widget.addTab(self.camera_widget, "Cameras")
        self.tab_widget.addTab(self.capture_volume_widget, "CaptureVolume")
        self.tab_widget.addTab(self.recording_widget, "Rec&ording")
        self.tab_widget.addTab(self.processing_widget, "&Processing")

        # default no cameras...can't record
        self.tab_widget.setTabEnabled(1, False)
        if not self.session.post_processing_eligible():
            self.tab_widget.setTabEnabled(2, False)
        
            
        old_index = self.tab_widget.currentIndex()

        # self.load_calibration_widget()
        # self.load_post_processing_widget()
        # cannot load recording widget until cameras are connected...
        # self.load_recording_widget()
        self.tab_widget.setCurrentIndex(old_index)
        self.connect_signals()

    def connect_signals(self):
        """
        After launching a session, connect signals and slots. 
        Much of these will be from the GUI to the session and vice-versa
        """
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.session.unlock_postprocessing.connect(self.load_post_processing_widget)
        

    # def load_calibration_widget(self):
    #     calibration_index = self.tab_widget.indexOf(self.calibration_widget)
    #     self.tab_widget.removeTab(calibration_index)
    #     self.calibration_widget.deleteLater()
    #     new_calibration_widget = CalibrationWidget(self.session)
    #     self.tab_widget.insertTab(
    #         calibration_index, new_calibration_widget, "&Calibration"
    #     )
    #     self.calibration_widget = new_calibration_widget

    def load_recording_widget(self):
        recording_index = self.tab_widget.indexOf(self.recording_widget)
        self.tab_widget.removeTab(recording_index)
        self.recording_widget.deleteLater()
        new_recording_widget = RecordingWidget(self.session)
        self.tab_widget.insertTab(recording_index, new_recording_widget, "Rec&ording")
        self.recording_widget = new_recording_widget

    def load_post_processing_widget(self):
        processing_index = self.tab_widget.indexOf(self.processing_widget)
        self.tab_widget.removeTab(processing_index)
        self.processing_widget.deleteLater()
        new_processing_widget = PostProcessingWidget(self.session)
        self.tab_widget.insertTab(
            processing_index, new_processing_widget, "&Processing"
        )
        self.processing_widget = new_processing_widget

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
    app = QApplication([])
    # log_widget = LogWidget()
    # log_widget.show()
    window = MainWindow()
    window.show()

    app.exec()


if __name__ == "__main__":
    launch_main()
