import logging
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path

import rtoml
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMenu,
    QTabWidget,
    QWidget,
)

from caliscope import APP_SETTINGS_PATH, LOG_DIR, __root__
from caliscope.cameras.camera_array import CameraArray
from caliscope.workspace_coordinator import WorkspaceCoordinator
from caliscope.task_manager import TaskHandle
from caliscope.gui.cameras_tab_widget import CamerasTabWidget
from caliscope.gui.charuco_widget import CharucoWidget
from caliscope.gui.log_widget import LogWidget
from caliscope.gui.reconstruction_tab import ReconstructionTab
from caliscope.gui.vizualize.calibration.capture_volume_visualizer import CaptureVolumeVisualizer
from caliscope.gui.extrinsic_calibration_tab import ExtrinsicCalibrationTab
from caliscope.gui.workspace_widget import WorkspaceSummaryWidget

logger = logging.getLogger(__name__)


class TabTypes(Enum):
    Workspace = 1
    Charuco = 2
    Cameras = 3
    CaptureVolume = 4


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.app_settings = rtoml.load(APP_SETTINGS_PATH)

        self.setWindowTitle("Caliscope")
        self.setWindowIcon(QIcon(str(Path(__root__, "caliscope/gui/icons/box3d-center.svg"))))
        self.setMinimumSize(500, 500)
        self.central_tab = QWidget(self)
        self.setCentralWidget(self.central_tab)

        self.build_menus()
        self.connect_menu_actions()
        self.build_docked_logger()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Graceful shutdown on app exit.

        Ensures all background threads and resources are cleaned up before
        the application terminates. Without this, threads spawned by the
        coordinator (TaskManager, streamers, synchronizers) would leak.
        """
        logger.info("Application exit initiated")

        # Clean up tabs that have presenter resources
        if hasattr(self, "cameras_tab_widget") and hasattr(self.cameras_tab_widget, "cleanup"):
            self.cameras_tab_widget.cleanup()
        if hasattr(self, "reconstruction_tab") and hasattr(self.reconstruction_tab, "cleanup"):
            self.reconstruction_tab.cleanup()

        # Coordinator cleanup (TaskManager shutdown)
        if hasattr(self, "coordinator"):
            self.coordinator.cleanup()

        logger.info("Application cleanup complete")
        super().closeEvent(event)

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
        self.workspace_summary = WorkspaceSummaryWidget(self.coordinator)
        self.workspace_summary.reload_workspace_btn.clicked.connect(self.reload_workspace)
        self.central_tab.addTab(self.workspace_summary, "Workspace")

        extrinsics_available = self.coordinator.all_extrinsic_mp4s_available()
        intrinsics_calibrated = self.coordinator.camera_array.all_intrinsics_calibrated()
        if extrinsics_available and intrinsics_calibrated:
            self.workspace_summary.calibrate_btn.setEnabled(True)
        else:
            self.workspace_summary.calibrate_btn.setEnabled(False)

        logger.info("Building Charuco widget")
        self.charuco_widget = CharucoWidget(self.coordinator)
        self.central_tab.addTab(self.charuco_widget, "Charuco")

        # Build Cameras tab with intrinsic calibration workflow
        if self.coordinator.cameras_loaded:
            logger.info("Building Cameras tab with intrinsic calibration")
            self.cameras_tab_widget = CamerasTabWidget(self.coordinator)
        else:
            logger.info("Cameras tab disabled - no intrinsic videos available")
            self.cameras_tab_widget = QWidget()
        self.central_tab.addTab(self.cameras_tab_widget, "Cameras")
        self.central_tab.setTabEnabled(
            self.find_tab_index_by_title("Cameras"),
            self.coordinator.cameras_loaded,
        )

        logger.info("About to load capture volume tab")
        if self.coordinator.capture_volume_loaded:
            logger.info("Creating ExtrinsicCalibrationTab")
            self.extrinsic_calibration_tab = ExtrinsicCalibrationTab(self.coordinator)
        else:
            logger.info("Creating dummy widget")
            self.extrinsic_calibration_tab = QWidget()
        self.central_tab.addTab(self.extrinsic_calibration_tab, "Capture Volume")
        self.central_tab.setTabEnabled(
            self.find_tab_index_by_title("Capture Volume"), self.coordinator.capture_volume_loaded
        )

        # Enable Capture Volume tab dynamically when extrinsic calibration completes
        self.coordinator.capture_volume_calibrated.connect(self._on_capture_volume_ready)

        # Track current tab for VTK suspend/resume (QTabWidget doesn't fire hideEvent)
        self._previous_tab_index: int = 0
        self.central_tab.currentChanged.connect(self._on_tab_changed)

        logger.info("About to load reconstruction tab")
        if self.coordinator.capture_volume_loaded and self.coordinator.recordings_available():
            logger.info("Creating reconstruction tab")
            presenter = self.coordinator.create_reconstruction_presenter()
            self.reconstruction_tab = ReconstructionTab(presenter)
            # Update camera array when coordinate system changes
            self.coordinator.capture_volume_shifted.connect(
                lambda: presenter.refresh_camera_array(self.coordinator.camera_array)
            )
            reconstruction_enabled = True
        else:
            logger.info("Creating dummy widget")
            self.reconstruction_tab = QWidget()
            reconstruction_enabled = False
        self.central_tab.addTab(self.reconstruction_tab, "Reconstruction")
        self.central_tab.setTabEnabled(self.find_tab_index_by_title("Reconstruction"), reconstruction_enabled)

    def build_docked_logger(self):
        # create log window which is fixed below main window
        self.docked_logger = QDockWidget("Log", self)
        self.docked_logger.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.docked_logger.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.log_widget = LogWidget()
        self.docked_logger.setWidget(self.log_widget)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docked_logger)

    def launch_workspace(self, path_to_workspace: str) -> TaskHandle:
        """Launch workspace and return TaskHandle for additional callbacks.

        Returns:
            TaskHandle for connecting additional completion callbacks.
        """
        logger.info(f"Launching session with config file stored in {path_to_workspace}")
        self.coordinator = WorkspaceCoordinator(Path(path_to_workspace))
        logger.info("Initiate coordinator loading")
        # TaskHandle.completed safe to connect after start - Qt queues cross-thread signals
        handle = self.coordinator.load_workspace()
        handle.completed.connect(self.build_central_tabs)

        self.open_project_action.setEnabled(False)
        self.open_recent_project_submenu.setEnabled(False)
        return handle

    def find_tab_index_by_title(self, title):
        # Iterate through tabs to find the index of the tab with the given title
        for index in range(self.central_tab.count()):
            if self.central_tab.tabText(index) == title:
                return index
        return -1  # Return -1 if the tab is not found

    def _on_capture_volume_ready(self) -> None:
        """Enable Capture Volume tab after extrinsic calibration completes."""
        idx = self.find_tab_index_by_title("Capture Volume")
        if idx < 0 or self.central_tab.isTabEnabled(idx):
            return  # Tab not found or already enabled

        logger.info("Enabling Capture Volume tab after calibration")

        # Replace dummy widget with real ExtrinsicCalibrationTab
        old_widget = self.central_tab.widget(idx)
        self.extrinsic_calibration_tab = ExtrinsicCalibrationTab(self.coordinator)
        self.central_tab.removeTab(idx)
        self.central_tab.insertTab(idx, self.extrinsic_calibration_tab, "Capture Volume")
        if old_widget is not None:
            old_widget.deleteLater()
        self.central_tab.setTabEnabled(idx, True)

    def _on_tab_changed(self, new_index: int) -> None:
        """Suspend/resume VTK rendering when switching tabs.

        QTabWidget doesn't fire hideEvent/showEvent on tab contents when switching
        tabs - it only stops painting them. VTK's interactor keeps polling for events,
        wasting CPU. We manually notify VTK widgets when their tab becomes inactive.
        """
        # Suspend VTK on previous tab if it has VTK
        prev_widget = self.central_tab.widget(self._previous_tab_index)
        if hasattr(prev_widget, "suspend_vtk"):
            logger.debug(f"Suspending VTK on tab {self._previous_tab_index}")
            prev_widget.suspend_vtk()

        # Resume VTK on new tab if it has VTK
        new_widget = self.central_tab.widget(new_index)
        if hasattr(new_widget, "resume_vtk"):
            logger.debug(f"Resuming VTK on tab {new_index}")
            new_widget.resume_vtk()

        self._previous_tab_index = new_index

    def reload_workspace(self):
        # Clear all existing tabs
        logger.info("Clearing workspace")
        # Iterate backwards through the tabs and remove them
        for index in range(self.central_tab.count() - 1, -1, -1):
            widget_to_remove = self.central_tab.widget(index)
            logger.info(f"Removing tab with index {index}")
            self.central_tab.removeTab(index)
            if widget_to_remove is not None:
                # Explicit cleanup for widgets that need it
                # (closeEvent not triggered by removeTab + deleteLater)
                if hasattr(widget_to_remove, "cleanup"):
                    widget_to_remove.cleanup()
                widget_to_remove.deleteLater()

            self.central_tab.clear()

        workspace = self.coordinator.workspace
        del self.coordinator
        self.coordinator = WorkspaceCoordinator(workspace_dir=workspace)
        handle = self.coordinator.load_workspace()
        handle.completed.connect(self.build_central_tabs)

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
        logger.info(f"Opening logging directory within File Explorer...  located at {LOG_DIR}")
        if sys.platform == "win32":
            os.startfile(LOG_DIR)
        elif sys.platform == "darwin":
            subprocess.run(["open", LOG_DIR])
        else:  # Linux and Unix-like systems
            subprocess.run(["xdg-open", LOG_DIR])
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
        logger.info(f"Saving out app settings to {APP_SETTINGS_PATH}")
        with open(APP_SETTINGS_PATH, "w") as f:
            rtoml.dump(self.app_settings, f)


def launch_main():
    # import qdarktheme

    app = QApplication(sys.argv)
    dummy_widget = CaptureVolumeVisualizer(camera_array=CameraArray({}))  #  try to force "blinking to initial main"
    del dummy_widget
    # qdarktheme.setup_theme("auto")
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    launch_main()
    # pass
