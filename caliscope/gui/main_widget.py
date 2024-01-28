import caliscope.logger
from pathlib import Path
from enum import Enum
import os
import sys
import subprocess
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QWidget,
    QTabWidget,
    QDockWidget,
    QMenu,
)
import rtoml
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt
from caliscope import __root__, __settings_path__
from caliscope.gui.log_widget import LogWidget
from caliscope.gui.charuco_widget import CharucoWidget
from caliscope.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget
from caliscope.gui.workspace_widget import WorkspaceSummaryWidget
from caliscope.gui.camera_management.multiplayback_widget import (
    MultiIntrinsicPlaybackWidget,
)
from caliscope.gui.post_processing_widget import PostProcessingWidget
from caliscope.controller import Controller
from caliscope import __log_dir__
from caliscope.gui.vizualize.calibration.capture_volume_visualizer import CaptureVolumeVisualizer
from caliscope.cameras.camera_array import CameraArray

logger = caliscope.logger.get(__name__)


class TabTypes(Enum):
    Workspace = 1
    Charuco = 2
    Cameras = 3
    CaptureVolume = 4


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.app_settings = rtoml.load(__settings_path__)

        self.setWindowTitle("Caliscope")
        self.setWindowIcon(QIcon(str(Path(__root__, "caliscope/gui/icons/box3d-center.svg"))))
        self.setMinimumSize(500, 500)
        self.central_tab = QWidget(self)
        self.setCentralWidget(self.central_tab)
        
        
        self.build_menus()
        self.connect_menu_actions()
        self.build_docked_logger()


    def connect_menu_actions(self):
        self.open_project_action.triggered.connect(self.create_new_project_folder)
        self.exit_pyxy3d_action.triggered.connect(QApplication.instance().quit)
        self.open_log_directory_action.triggered.connect(self.open_log_dir)

    def build_menus(self):
        # File Menu
        self.menu = self.menuBar()

        # CREATE FILE MENU
        self.file_menu = self.menu.addMenu("&File")
        self.open_project_action = QAction("New/Open Project", self)
        self.file_menu.addAction(self.open_project_action)

        ####################  Open Recent  ################################
        self.open_recent_project_submenu = QMenu("Recent Projects...", self)

        # Populate the submenu with recent project paths;
        # reverse so that last one appended is at the top of the list
        for project_path in reversed(self.app_settings["recent_projects"]):
            self.add_to_recent_project(project_path)

        self.file_menu.addMenu(self.open_recent_project_submenu)
        ###################################################################

        self.open_log_directory_action = QAction("Open Log Directory")
        self.file_menu.addAction(self.open_log_directory_action)
        self.exit_pyxy3d_action = QAction("Exit", self)
        self.file_menu.addAction(self.exit_pyxy3d_action)

    def build_central_tabs(self):
        self.central_tab = QTabWidget(self)
        self.setCentralWidget(self.central_tab)

        logger.info("Building workspace summary")
        self.workspace_summary = WorkspaceSummaryWidget(self.controller)
        self.workspace_summary.reload_workspace_btn.clicked.connect(self.reload_workspace)
        self.central_tab.addTab(self.workspace_summary, "Workspace")

        if (
            self.controller.all_extrinsic_mp4s_available()
            and self.controller.camera_array.all_intrinsics_calibrated()
        ):
            self.workspace_summary.calibrate_btn.setEnabled(True)
        else:
            self.workspace_summary.calibrate_btn.setEnabled(False)

        logger.info("Building Charuco widget")
        self.charuco_widget = CharucoWidget(self.controller)
        self.central_tab.addTab(self.charuco_widget, "Charuco")

        logger.info("About to load Camera tab")
        if self.controller.cameras_loaded:
            logger.info("Creating MultiIntrinsic Playback Widget")
            self.intrinsic_cal_widget = MultiIntrinsicPlaybackWidget(self.controller)
            logger.info("MultiIntrinsic Playback Widget created")
        else:
            self.intrinsic_cal_widget = QWidget()

        logger.info("finished loading camera tab")
        self.central_tab.addTab(self.intrinsic_cal_widget, "Cameras")
        self.central_tab.setTabEnabled(
            self.find_tab_index_by_title("Cameras"), self.controller.cameras_loaded
        )
        logger.info("Camera tab enabled")

        logger.info("About to load capture volume tab")
        if self.controller.capture_volume_loaded:
            logger.info("Creating capture Volume Widget")
            self.capture_volume_widget = CaptureVolumeWidget(self.controller)
        else:
            logger.info("Creating dummy widget")
            self.capture_volume_widget = QWidget()
        self.central_tab.addTab(self.capture_volume_widget, "Capture Volume")
        self.central_tab.setTabEnabled(
            self.find_tab_index_by_title("Capture Volume"), self.controller.capture_volume_loaded
        )

        logger.info("About to load post-processing tab")
        if self.controller.capture_volume_loaded and self.controller.recordings_available():
            logger.info("Creating post processing widget")
            self.post_processing_widget = PostProcessingWidget(self.controller)
            self.controller.capture_volume_shifted.connect(self.post_processing_widget.refresh_visualizer)
            post_processing_enabled = True
        else:
            logger.info("Creating dummy widget")
            self.post_processing_widget = QWidget()
            post_processing_enabled = False
        self.central_tab.addTab(self.post_processing_widget, "Post Processing")
        self.central_tab.setTabEnabled(
            self.find_tab_index_by_title("Post Processing"), post_processing_enabled
        )

    def build_docked_logger(self):
        # create log window which is fixed below main window
        self.docked_logger = QDockWidget("Log", self)
        self.docked_logger.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.docked_logger.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.log_widget = LogWidget()
        self.docked_logger.setWidget(self.log_widget)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docked_logger)

    def launch_workspace(self, path_to_workspace: str):
        logger.info(f"Launching session with config file stored in {path_to_workspace}")
        self.controller = Controller(Path(path_to_workspace))
        self.controller.load_workspace_thread.finished.connect(self.build_central_tabs)
        logger.info("Initiate controller loading")
        self.controller.load_workspace()

        self.open_project_action.setEnabled(False)
        self.open_recent_project_submenu.setEnabled(False)


    def find_tab_index_by_title(self, title):
        # Iterate through tabs to find the index of the tab with the given title
        for index in range(self.central_tab.count()):
            if self.central_tab.tabText(index) == title:
                return index
        return -1  # Return -1 if the tab is not found

    def reload_workspace(self):
        # Clear all existing tabs
        logger.info("Clearing workspace")
        # Iterate backwards through the tabs and remove them
        for index in range(self.central_tab.count() - 1, -1, -1):
            widget_to_remove = self.central_tab.widget(index)
            logger.info(f"Removing tab with index {index}")
            self.central_tab.removeTab(index)
            if widget_to_remove is not None:
                widget_to_remove.deleteLater()

            self.central_tab.clear()

        workspace = self.controller.workspace
        del self.controller
        self.controller = Controller(workspace_dir=workspace)
        self.controller.load_workspace()
        self.controller.load_workspace_thread.finished.connect(self.build_central_tabs)
        

    def add_to_recent_project(self, project_path: str):
        recent_project_action = QAction(project_path, self)
        recent_project_action.triggered.connect(self.open_recent_project)
        self.open_recent_project_submenu.addAction(recent_project_action)

    def open_recent_project(self):
        action = self.sender()
        project_path = action.text()
        logger.info(f"Opening recent session stored at {project_path}")
        self.launch_workspace(project_path)

    def open_log_dir(self):
        logger.info(f"Opening logging directory within File Explorer...  located at {__log_dir__}")
        if sys.platform == 'win32':
            os.startfile(__log_dir__)
        elif sys.platform == 'darwin':
            subprocess.run(["open", __log_dir__])
        else:  # Linux and Unix-like systems
            subprocess.run(["xdg-open", __log_dir__])
        pass
        
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
        logger.info(f"Saving out app settings to {__settings_path__}")
        with open(__settings_path__, "w") as f:
            rtoml.dump(self.app_settings, f)


def launch_main():
    import qdarktheme

    app = QApplication(sys.argv)
    dummy_widget = CaptureVolumeVisualizer(camera_array=CameraArray({})) #  try to force "blinking to initial main"
    del dummy_widget
    qdarktheme.setup_theme("auto")
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    launch_main()
    # pass
