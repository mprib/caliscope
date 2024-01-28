import caliscope.logger
from pathlib import Path


from PySide6.QtWidgets import QMainWindow, QFileDialog
from threading import Thread
import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QDockWidget,
    QMenu,
)
import rtoml
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt
from caliscope import __root__, __settings_path__
from caliscope.session.session import LiveSession, SessionMode
from caliscope.gui.log_widget import LogWidget
from caliscope.configurator import Configurator
from caliscope.gui.charuco_widget import CharucoWidget
from caliscope.gui.live_camera_config.intrinsic_calibration_widget import (
    IntrinsicCalibrationWidget,
)
from caliscope.gui.recording_widget import RecordingWidget
from caliscope.gui.post_processing_widget import PostProcessingWidget
from caliscope.gui.extrinsic_calibration_widget import ExtrinsicCalibrationWidget
from caliscope.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget

logger = caliscope.logger.get(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.app_settings = rtoml.load(__settings_path__)

        self.setWindowTitle("Caliscope")
        self.setWindowIcon(QIcon(str(Path(__root__, "caliscope/gui/icons/pyxy_logo.svg"))))
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
        self.triangulate_mode_select = QAction(SessionMode.Triangulate.value)
        self.mode_menu.addAction(self.charuco_mode_select)
        self.mode_menu.addAction(self.intrinsic_mode_select)
        self.mode_menu.addAction(self.extrinsic_mode_select)
        self.mode_menu.addAction(self.capture_volume_mode_select)
        self.mode_menu.addAction(self.recording_mode_select)
        self.mode_menu.addAction(self.triangulate_mode_select)

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

        # create a reverse lookup dictionary to pull the mode enum that should be activated
        SessionModeLookup = {mode.value: mode for mode in SessionMode}
        mode = SessionModeLookup[action.text()]
        logger.info(f"Attempting to set session mode to {mode.value}")
        self.session.set_mode(mode)
        logger.info(f"Successful change to {mode} Mode")

    def update_central_widget_mode(self):
        """
        This will be triggered whenever the session successfully completes a mode change and emits
        a signal to that effect.
        """
        logger.info("Begin process of updating central widget")

        old_widget = self.centralWidget()
        self.setCentralWidget(QWidget())
        old_widget.deleteLater()

        logger.info("Clearing events in emmitter threads to get them to wind down")
        if type(old_widget) == RecordingWidget:
            old_widget.thumbnail_emitter.keep_collecting.clear()
            logger.info("Waiting for recording widget to wrap up")
            old_widget.thumbnail_emitter.wait()

        if type(old_widget) == ExtrinsicCalibrationWidget:
            old_widget.paired_frame_emitter.keep_collecting.clear()
            logger.info("Waiting for extrinsic calibration widget to wrap up")
            old_widget.paired_frame_emitter.wait()

        if type(old_widget) == IntrinsicCalibrationWidget:
            for port, tab in old_widget.camera_tabs.tab_widgets.items():
                tab.frame_emitter.keep_collecting.clear()

        logger.info(f"Matching next tab to active session mode: {self.session.mode}")
        # Create the new central widget based on the mode
        match self.session.mode:
            case SessionMode.Charuco:
                new_widget = CharucoWidget(self.session)
            case SessionMode.IntrinsicCalibration:
                new_widget = IntrinsicCalibrationWidget(self.session)
            case SessionMode.ExtrinsicCalibration:
                logger.info("About to create extrinsic calibration widget")
                new_widget = ExtrinsicCalibrationWidget(self.session)
            case SessionMode.CaptureVolumeOrigin:
                new_widget = CaptureVolumeWidget(self.session)
            case SessionMode.Recording:
                new_widget = RecordingWidget(self.session)
            case SessionMode.Triangulate:
                new_widget = PostProcessingWidget(self.session)

        self.setCentralWidget(new_widget)

    def switch_to_capture_volume(self):
        """
        Once the extrinsic calibration is complete, the GUI should automatically switch over to the capture volume widget
        """
        self.session.set_mode(SessionMode.CaptureVolumeOrigin)

    def update_enable_disable(self):
        # note: if the cameras are connected,then you can peak
        # into extrinsic/recording tabs, though cannot collect data

        # you can always look at a charuco board
        self.charuco_mode_select.setEnabled(True)

        if self.session.is_camera_setup_eligible():
            self.intrinsic_mode_select.setEnabled(True)
            self.extrinsic_mode_select.setEnabled(True)
            self.recording_mode_select.setEnabled(True)
        else:
            self.intrinsic_mode_select.setEnabled(False)
            self.extrinsic_mode_select.setEnabled(False)
            self.recording_mode_select.setEnabled(False)

        if self.session.is_capture_volume_eligible():
            self.capture_volume_mode_select.setEnabled(True)
        else:
            self.capture_volume_mode_select.setEnabled(False)

        if self.session.is_triangulate_eligible():
            self.triangulate_mode_select.setEnabled(True)
        else:
            self.triangulate_mode_select.setEnabled(False)

    def disconnect_cameras(self):
        self.session.set_mode(SessionMode.Charuco)
        self.session.disconnect_cameras()
        self.disconnect_cameras_action.setEnabled(False)
        self.connect_cameras_action.setEnabled(True)
        self.update_enable_disable()

    def pause_all_frame_reading(self):
        logger.info(
            "Pausing all frame reading at load of stream tools; should be on charuco tab right now"
        )
        self.session.pause_all_monocalibrators()
        self.session.pause_synchronizer()

    def load_stream_tools(self):
        self.connect_cameras_action.setEnabled(False)
        self.disconnect_cameras_action.setEnabled(True)
        self.session.qt_signaler.stream_tools_loaded_signal.connect(
            self.pause_all_frame_reading
        )
        self.thread = Thread(
            target=self.session.load_stream_tools, args=(), daemon=True
        )
        self.thread.start()

    def launch_session(self, path_to_folder: str):
        session_path = Path(path_to_folder)
        self.config = Configurator(session_path)
        logger.info(f"Launching session with config file stored in {session_path}")
        self.session = LiveSession(self.config)

        # can always load charuco
        self.charuco_widget = CharucoWidget(self.session)
        self.setCentralWidget(self.charuco_widget)

        # now connecting to cameras is an option
        self.connect_cameras_action.setEnabled(True)

        # but must exit and start over to launch a new session for now
        self.connect_session_signals()

        self.open_project_action.setEnabled(False)
        self.open_recent_project_submenu.setEnabled(False)
        self.update_enable_disable()

    def connect_session_signals(self):
        """
        After launching a session, connect signals and slots.
        Much of these will be from the GUI to the session and vice-versa
        """
        self.session.qt_signaler.unlock_postprocessing.connect(
            self.update_enable_disable
        )
        self.session.qt_signaler.mode_change_success.connect(
            self.update_central_widget_mode
        )
        self.session.qt_signaler.stream_tools_loaded_signal.connect(
            self.update_enable_disable
        )
        self.session.qt_signaler.stream_tools_disconnected_signal.connect(
            self.update_enable_disable
        )
        self.session.qt_signaler.mode_change_success.connect(self.update_enable_disable)
        self.session.qt_signaler.extrinsic_calibration_complete.connect(
            self.switch_to_capture_volume
        )

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
            dir=str(default_folder),
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
            rtoml.dump(self.app_settings, f)


def launch_main():
    import qdarktheme

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("auto")
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    launch_main()
