
import pyxy3d.logger
from pathlib import Path

import sys
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QWidget,
    QTabWidget,
    QDockWidget,
    QMenu,
)
import toml
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt
from pyxy3d import __root__, __settings_path__
from pyxy3d.gui.log_widget import LogWidget
from pyxy3d.gui.charuco_widget import CharucoWidget
from pyxy3d.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget
from pyxy3d.gui.workspace_widget import WorkspaceSummaryWidget
from pyxy3d.gui.prerecorded_intrinsic_calibration.multiplayback_widget import MultiIntrinsicPlaybackWidget
from pyxy3d.controller import Controller

logger = pyxy3d.logger.get(__name__)


class PreRecordedMainWindow(QMainWindow):
    def __init__(self):
        super(PreRecordedMainWindow, self).__init__()

        self.app_settings = toml.load(__settings_path__)

        self.setWindowTitle("Pyxy3D")
        self.setWindowIcon(QIcon(str(Path(__root__, "pyxy3d/gui/icons/pyxy_logo.svg"))))
        self.setMinimumSize(500, 500)

        self.build_menus()
        self.build_docked_logger()

    def connect_menu_actions(self):
        self.open_project_action.triggered.connect(self.create_new_project_folder)
        self.exit_pyxy3d_action.triggered.connect(QApplication.instance().quit)

    def build_menus(self):
        # File Menu
        self.menu = self.menuBar()

        # CREATE FILE MENU
        self.file_menu = self.menu.addMenu("&File")
        self.open_project_action = QAction("New/Open Project", self)
        self.file_menu.addAction(self.open_project_action)

        self.calibrate_capture_volume = QAction("Calibrate Capture Volume", self)
        self.file_menu.addAction(self.calibrate_capture_volume)

        # Open Recent
        self.open_recent_project_submenu = QMenu("Recent Projects...", self)

        # Populate the submenu with recent project paths;
        # reverse so that last one appended is at the top of the list
        for project_path in reversed(self.app_settings["recent_projects"]):
            self.add_to_recent_project(project_path)

        self.file_menu.addMenu(self.open_recent_project_submenu)

        self.exit_pyxy3d_action = QAction("Exit", self)
        self.file_menu.addAction(self.exit_pyxy3d_action)

    def build_central_tabs(self):
    
        self.central_tab = QTabWidget()
        self.setCentralWidget(self.central_tab)
        self.workspace_summary = WorkspaceSummaryWidget(self.controller)
        self.central_tab.addTab(self.workspace_summary, "Workspace")

        self.charuco_widget = CharucoWidget(self.controller)
        self.central_tab.addTab(self.charuco_widget,"Charuco")    
        self.intrinsic_cal_widget = MultiIntrinsicPlaybackWidget(self.controller)
        self.central_tab.addTab(self.intrinsic_cal_widget, "Cameras") 
        self.capture_volume_widget = CaptureVolumeWidget(self.controller)
        self.central_tab.addTab(self.capture_volume_widget, "Capture Volume")

    def build_docked_logger(self):
        # create log window which is fixed below main window
        self.docked_logger = QDockWidget("Log", self)
        self.docked_logger.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.docked_logger.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.log_widget = LogWidget()
        self.docked_logger.setWidget(self.log_widget)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docked_logger)
        
                
    def update_enable_disable(self):
        # note: if the cameras are connected,then you can peak
        # into extrinsic/recording tabs, though cannot collect data

        # you can always look at a charuco board
        # self.charuco_mode_select.setEnabled(True)
        pass # this may be useful later so leaving it here as a template

        # the code below gives a sense of how I previously managed this
        # if self.session.is_camera_setup_eligible():
        #     self.intrinsic_mode_select.setEnabled(True)
        #     self.extrinsic_mode_select.setEnabled(True)
        #     self.recording_mode_select.setEnabled(True)
        # else:
        #     self.intrinsic_mode_select.setEnabled(False)
        #     self.extrinsic_mode_select.setEnabled(False)
        #     self.recording_mode_select.setEnabled(False)

        # if self.session.is_capture_volume_eligible():
        #     self.capture_volume_mode_select.setEnabled(True)
        # else:
        #     self.capture_volume_mode_select.setEnabled(False)

        # if self.session.is_triangulate_eligible():
        #     self.triangulate_mode_select.setEnabled(True)
        # else:
        #     self.triangulate_mode_select.setEnabled(False)

    def launch_workspace(self, path_to_workspace: str):
        logger.info(f"Launching session with config file stored in {path_to_workspace}")
        self.controller = Controller(Path(path_to_workspace))
        self.controller.load_camera_array()
        self.controller.load_intrinsic_streams()
        self.controller.load_estimated_capture_volume()
        # must have controller in
        self.build_central_tabs()

        # but must exit and start over to launch a new session for now
        self.connect_controller_signals()

        self.open_project_action.setEnabled(False)
        self.open_recent_project_submenu.setEnabled(False)
        self.update_enable_disable()

    def connect_controller_signals(self):
        """
        After launching a session, connect signals and slots.
        Much of these will be from the GUI to the session and vice-versa
        """
        pass
        # some placeholder code that might get implemented:
        # self.controller.unlock_postprocessing.connect(
        #     self.update_enable_disable
        # )
        # self.controller.extrinsic_calibration_complete.connect(
        #     self.switch_to_capture_volume
        # )

    def add_to_recent_project(self, project_path: str):
        recent_project_action = QAction(project_path, self)
        recent_project_action.triggered.connect(self.open_recent_project)
        self.open_recent_project_submenu.addAction(recent_project_action)

    def open_recent_project(self):
        action = self.sender()
        project_path = action.text()
        logger.info(f"Opening recent session stored at {project_path}")
        self.launch_workspace(project_path)

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
            self.launch_workspace(path_to_folder)

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
    import qdarktheme

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("auto")
    window = PreRecordedMainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    # launch_main()
    pass
