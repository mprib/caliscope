
import pyxy3d.logger
from pathlib import Path
from enum import Enum

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
from pyxy3d.gui.camera_management.multiplayback_widget import MultiIntrinsicPlaybackWidget
from pyxy3d.gui.post_processing_widget import PostProcessingWidget
from pyxy3d.controller import Controller

logger = pyxy3d.logger.get(__name__)

class TabTypes(Enum):
    Workspace = 1
    Charuco = 2
    Cameras = 3
    CaptureVolume = 4

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

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

        self.reload_workspace_action = QAction("Reload workspace", self)
        self.file_menu.addAction(self.reload_workspace_action)

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
        if not hasattr(self, "central_tab"): 
            self.central_tab = QTabWidget()
            self.setCentralWidget(self.central_tab)

        logger.info("Building workspace summary")
        self.workspace_summary = WorkspaceSummaryWidget(self.controller)
        self.central_tab.addTab(self.workspace_summary, "Workspace")

        logger.info("Building Charuco widget")
        self.charuco_widget = CharucoWidget(self.controller)
        self.central_tab.addTab(self.charuco_widget,"Charuco")    
        

        logger.info("About to load Camera tab")
        if self.controller.all_instrinsic_mp4s():
            logger.info("Loading intrinsic stream manager")
            self.controller.load_camera_array()
            self.controller.load_intrinsic_stream_manager()
            logger.info("Creating MultiIntrinsic Playback Widget")
            self.intrinsic_cal_widget = MultiIntrinsicPlaybackWidget(self.controller)
            logger.info("MultiIntrinsic Playback Widget created")
            cameras_enabled = True
        else:
            self.intrinsic_cal_widget = QWidget()
            cameras_enabled = False
        
        logger.info("finished loading camera tab")
        self.central_tab.addTab(self.intrinsic_cal_widget, "Cameras")
        self.central_tab.setTabEnabled(self.find_tab_index_by_title("Cameras"),cameras_enabled)
        logger.info("Camera tab enabled")

        logger.info("About to load capture volume tab")
        if self.controller.all_extrinsics_estimated():
            self.controller.load_estimated_capture_volume()
            logger.info("Creating capture Volume Widget")
            self.capture_volume_widget = CaptureVolumeWidget(self.controller)
            capture_volume_enabled = True
        else:
            self.capture_volume_widget = QWidget()
            capture_volume_enabled = False
        self.central_tab.addTab(self.capture_volume_widget, "Capture Volume")
        self.central_tab.setTabEnabled(self.find_tab_index_by_title("Capture Volume"),capture_volume_enabled)

        
        logger.info("About to load post-processing tab")
        if self.controller.recordings_available():
            self.post_processing_widget = PostProcessingWidget(self.controller)
            post_processing_enabled = True
        else:
            self.post_processing_widget = QWidget()
            post_processing_enabled = False
        self.central_tab.addTab(self.post_processing_widget, "Post Processing")
        self.central_tab.setTabEnabled(self.find_tab_index_by_title("Post Processing"),post_processing_enabled)
        
    def build_docked_logger(self):
        # create log window which is fixed below main window
        self.docked_logger = QDockWidget("Log", self)
        self.docked_logger.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.docked_logger.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.log_widget = LogWidget()
        self.docked_logger.setWidget(self.log_widget)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docked_logger)
        
     
    def update_enable_disable(self):
        if self.controller.all_instrinsic_mp4s():
            self.central_tab.setTabEnabled(self.find_tab_index_by_title("Cameras"),True)
        else:
            self.central_tab.setTabEnabled(self.find_tab_index_by_title("Cameras"),False)

        if self.controller.all_extrinsic_mp4s() and self.controller.camera_array.all_intrinsics_calibrated():
            self.workspace_summary.calibrate_btn.setEnabled(True)
        else:
            self.workspace_summary.calibrate_btn.setEnabled(False)



    def launch_workspace(self, path_to_workspace: str):
        logger.info(f"Launching session with config file stored in {path_to_workspace}")
        self.controller = Controller(Path(path_to_workspace))
        
        self.build_central_tabs()

        # but must exit and start over to launch a new session for now
        self.connect_controller_signals()

        self.open_project_action.setEnabled(False)
        self.open_recent_project_submenu.setEnabled(False)
        self.update_enable_disable()

    def connect_controller_signals(self):
        self.reload_workspace_action.triggered.connect(self.reload_workspace)

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
            self.central_tab.removeTab(index)
            if widget_to_remove is not None:
                widget_to_remove.deleteLater()
        
            self.central_tab.clear()
    
        if hasattr(self.controller, "intrinsic_stream_manager"):
            logger.info("Attempting to wind down currently existing stream tools")
            self.controller.intrinsic_stream_manager.close_stream_tools()            

        # Rebuild the central tabs
        logger.info("Building Central tabs")
        self.build_central_tabs()

        # Update any necessary states or enable/disable UI elements
        self.update_enable_disable() 
        
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
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    # launch_main()
    pass
