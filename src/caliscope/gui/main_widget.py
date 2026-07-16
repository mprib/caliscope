from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import rtoml
from PySide6.QtCore import QTimer, Qt
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

from caliscope import APP_SETTINGS_PATH, LOG_DIR
from caliscope.gui import ICONS_DIR
from caliscope.gui.widgets.welcome_widget import WelcomeWidget

if TYPE_CHECKING:
    from caliscope.task_manager import TaskHandle

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.app_settings = rtoml.load(APP_SETTINGS_PATH)

        self.setWindowTitle("Caliscope")
        self.setWindowIcon(QIcon(str(ICONS_DIR / "box3d-center.svg")))
        self.setMinimumSize(500, 500)

        self.build_menus()
        self.connect_menu_actions()

        welcome = WelcomeWidget(self.recent_projects())
        welcome.open_project_requested.connect(self.create_new_project_folder)
        welcome.recent_project_selected.connect(self.launch_workspace)
        self.setCentralWidget(welcome)

    def recent_projects(self) -> list[str]:
        """Newest-first recent project paths whose directories still exist."""
        return [p for p in reversed(self.app_settings["recent_projects"]) if Path(p).is_dir()]

    def closeEvent(self, event: QCloseEvent) -> None:
        """Graceful shutdown on app exit.

        Ensures all background threads and resources are cleaned up before
        the application terminates. Without this, threads spawned by the
        coordinator (TaskManager, streamers, synchronizers) would leak.
        """
        logger.info("Application exit initiated")

        from caliscope.gui.cameras_tab_widget import CamerasTabWidget
        from caliscope.gui.multi_camera_processing_tab import MultiCameraProcessingTab
        from caliscope.gui.reconstruction_tab import ReconstructionTab

        cameras = getattr(self, "cameras_tab_widget", None)
        if isinstance(cameras, CamerasTabWidget):
            cameras.cleanup()
        multi = getattr(self, "multi_camera_tab", None)
        if isinstance(multi, MultiCameraProcessingTab):
            multi.cleanup()
        recon = getattr(self, "reconstruction_tab", None)
        if isinstance(recon, ReconstructionTab):
            recon.cleanup()

        # Coordinator cleanup (TaskManager shutdown)
        if hasattr(self, "coordinator"):
            self.coordinator.cleanup()

        logger.info("Application cleanup complete")
        super().closeEvent(event)

    def connect_menu_actions(self):
        self.open_project_action.triggered.connect(self.create_new_project_folder)
        app = QApplication.instance()
        assert app is not None
        self.exit_pyxy3d_action.triggered.connect(app.quit)
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

        for project_path in self.recent_projects():
            self.add_to_recent_project(project_path)

        self.file_menu.addMenu(self.open_recent_project_submenu)
        ###################################################################

        self.open_log_directory_action = QAction("Open Log Directory")
        self.file_menu.addAction(self.open_log_directory_action)
        self.exit_pyxy3d_action = QAction("Exit", self)
        self.file_menu.addAction(self.exit_pyxy3d_action)

    def build_central_tabs(self, _result: object = None) -> None:
        self.central_tab = QTabWidget(self)
        self.setCentralWidget(self.central_tab)

        self.statusBar().showMessage("Building Project tab…")

        from caliscope.gui.views.project_setup_view import ProjectSetupView

        logger.info("Building Project setup tab")
        self.project_tab = ProjectSetupView(self.coordinator)
        self.project_tab.tab_navigation_requested.connect(self._navigate_to_tab)
        self.central_tab.addTab(self.project_tab, "Project")

        self.coordinator.status_changed.connect(self._refresh_tab_enablement)
        self._previous_tab_index: int = 0
        self.central_tab.currentChanged.connect(self._on_tab_changed)
        self.open_project_action.setEnabled(True)
        self.open_recent_project_submenu.setEnabled(True)

        # Remaining tabs added one per event loop tick so the UI stays responsive
        self._deferred_tab_builders: deque[tuple[str, Callable[[], None]]] = deque(
            [
                ("Cameras", self._add_cameras_tab),
                ("Multi-Camera", self._add_multi_camera_tab),
                ("Calibrate", self._add_calibrate_tab),
                ("Reconstruction", self._add_reconstruction_tab),
            ]
        )
        QTimer.singleShot(0, self._build_next_deferred_tab)

    def _build_next_deferred_tab(self) -> None:
        if not self._deferred_tab_builders:
            self.statusBar().showMessage("Ready", 3000)
            return
        label, builder = self._deferred_tab_builders.popleft()
        self.statusBar().showMessage(f"Building {label} tab…")
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        builder()
        if self._deferred_tab_builders:
            QTimer.singleShot(0, self._build_next_deferred_tab)
        else:
            self.statusBar().showMessage("Ready", 3000)

    def _add_cameras_tab(self) -> None:
        from caliscope.gui.cameras_tab_widget import CamerasTabWidget
        from caliscope.gui.widgets.cameras_info_placeholder import CamerasInfoPlaceholder

        if self.coordinator.cameras_tab_enabled:
            logger.info("Building Cameras tab with intrinsic calibration")
            self.cameras_tab_widget: QWidget = CamerasTabWidget(self.coordinator)
        else:
            logger.info("No intrinsic videos - Cameras tab shows skip-intrinsics placeholder")
            self.cameras_tab_widget = CamerasInfoPlaceholder()
        self.central_tab.addTab(self.cameras_tab_widget, "Cameras")

    def _add_multi_camera_tab(self) -> None:
        from caliscope.gui.multi_camera_processing_tab import MultiCameraProcessingTab

        enabled = self.coordinator.multi_camera_tab_enabled
        if enabled:
            logger.info("Building Multi-Camera processing tab")
            self.multi_camera_tab: QWidget = MultiCameraProcessingTab(self.coordinator)
        else:
            self.multi_camera_tab = QWidget()
        self.central_tab.addTab(self.multi_camera_tab, "Multi-Camera")
        self.central_tab.setTabEnabled(self.find_tab_index_by_title("Multi-Camera"), enabled)

    def _add_calibrate_tab(self) -> None:
        from caliscope.gui.extrinsic_calibration_tab import ExtrinsicCalibrationTab

        enabled = self.coordinator.capture_volume_tab_enabled
        if enabled:
            logger.info("Creating ExtrinsicCalibrationTab")
            self.extrinsic_calibration_tab: QWidget = ExtrinsicCalibrationTab(self.coordinator)
            self.extrinsic_calibration_tab.navigation_requested.connect(self._navigate_to_tab)  # type: ignore[union-attr]
        else:
            self.extrinsic_calibration_tab = QWidget()
        self.central_tab.addTab(self.extrinsic_calibration_tab, "Calibrate")
        self.central_tab.setTabEnabled(self.find_tab_index_by_title("Calibrate"), enabled)

    def _add_reconstruction_tab(self) -> None:
        from caliscope.gui.reconstruction_tab import ReconstructionTab

        enabled = self.coordinator.reconstruction_tab_enabled
        if enabled:
            logger.info("Creating reconstruction tab")
            presenter = self.coordinator.create_reconstruction_presenter()
            self.reconstruction_tab: QWidget = ReconstructionTab(presenter)
            self.coordinator.capture_volume_updated.connect(
                lambda: presenter.refresh_camera_array(self.coordinator.camera_array)
            )
        else:
            self.reconstruction_tab = QWidget()
        self.central_tab.addTab(self.reconstruction_tab, "Reconstruction")
        self.central_tab.setTabEnabled(self.find_tab_index_by_title("Reconstruction"), enabled)

    def _replace_placeholder_tab(self, tab_name: str) -> None:
        from caliscope.gui.cameras_tab_widget import CamerasTabWidget
        from caliscope.gui.extrinsic_calibration_tab import ExtrinsicCalibrationTab
        from caliscope.gui.multi_camera_processing_tab import MultiCameraProcessingTab
        from caliscope.gui.reconstruction_tab import ReconstructionTab
        from caliscope.gui.widgets.cameras_info_placeholder import CamerasInfoPlaceholder

        idx = self.find_tab_index_by_title(tab_name)
        if idx < 0:
            return
        old = self.central_tab.widget(idx)

        if tab_name == "Cameras":
            if isinstance(old, CamerasTabWidget):
                return
            if self.coordinator.cameras_tab_enabled:
                logger.info("Building Cameras tab with intrinsic calibration")
                self.cameras_tab_widget = CamerasTabWidget(self.coordinator)
            else:
                logger.info("No intrinsic videos - Cameras tab shows skip-intrinsics placeholder")
                self.cameras_tab_widget = CamerasInfoPlaceholder()
            self.central_tab.removeTab(idx)
            self.central_tab.insertTab(idx, self.cameras_tab_widget, "Cameras")

        elif tab_name == "Multi-Camera":
            if isinstance(old, MultiCameraProcessingTab):
                return
            logger.info("Building Multi-Camera processing tab")
            self.multi_camera_tab = MultiCameraProcessingTab(self.coordinator)
            self.central_tab.removeTab(idx)
            self.central_tab.insertTab(idx, self.multi_camera_tab, "Multi-Camera")

        elif tab_name == "Calibrate":
            if isinstance(old, ExtrinsicCalibrationTab):
                return
            logger.info("Creating ExtrinsicCalibrationTab")
            self.extrinsic_calibration_tab = ExtrinsicCalibrationTab(self.coordinator)
            self.extrinsic_calibration_tab.navigation_requested.connect(self._navigate_to_tab)
            self.central_tab.removeTab(idx)
            self.central_tab.insertTab(idx, self.extrinsic_calibration_tab, "Calibrate")

        elif tab_name == "Reconstruction":
            if isinstance(old, ReconstructionTab):
                return
            logger.info("Creating reconstruction tab")
            presenter = self.coordinator.create_reconstruction_presenter()
            self.reconstruction_tab = ReconstructionTab(presenter)
            self.coordinator.capture_volume_updated.connect(
                lambda: presenter.refresh_camera_array(self.coordinator.camera_array)
            )
            self.central_tab.removeTab(idx)
            self.central_tab.insertTab(idx, self.reconstruction_tab, "Reconstruction")

        if old is not None:
            old.deleteLater()

    def _refresh_tab_enablement(self) -> None:
        """Refresh tab enabled states from computed properties.

        Called when Coordinator.status_changed fires (filesystem change,
        calibration complete, etc.).
        """
        # Update enabled state for each tab. The Cameras tab is exempt: it
        # stays enabled and shows a placeholder until intrinsic videos exist.
        self.central_tab.setTabEnabled(
            self.find_tab_index_by_title("Multi-Camera"),
            self.coordinator.multi_camera_tab_enabled,
        )
        self.central_tab.setTabEnabled(
            self.find_tab_index_by_title("Calibrate"),
            self.coordinator.capture_volume_tab_enabled,
        )
        self.central_tab.setTabEnabled(
            self.find_tab_index_by_title("Reconstruction"),
            self.coordinator.reconstruction_tab_enabled,
        )

        # If a tab became enabled and has a dummy widget, replace it
        self._maybe_replace_dummy_tabs()

    def _maybe_replace_dummy_tabs(self) -> None:
        """Replace dummy widgets with real tabs when they become enabled."""
        if self.coordinator.cameras_tab_enabled:
            self._replace_placeholder_tab("Cameras")
        if self.coordinator.multi_camera_tab_enabled:
            self._replace_placeholder_tab("Multi-Camera")
        if self.coordinator.capture_volume_tab_enabled:
            self._replace_placeholder_tab("Calibrate")
        if self.coordinator.reconstruction_tab_enabled:
            self._replace_placeholder_tab("Reconstruction")

    def build_docked_logger(self):
        from caliscope.gui.log_widget import LogWidget

        self.docked_logger = QDockWidget("Log", self)
        self.docked_logger.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.docked_logger.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.log_widget = LogWidget()
        self.docked_logger.setWidget(self.log_widget)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docked_logger)

    def launch_workspace(self, path_to_workspace: str) -> TaskHandle | None:
        """Launch workspace and return TaskHandle for additional callbacks."""
        # Tear down existing coordinator to avoid orphaned thread pools
        if hasattr(self, "coordinator"):
            self.coordinator.cleanup()
            del self.coordinator

        # Show loading state on welcome widget and flush the paint
        # before the blocking WorkspaceCoordinator constructor
        central = self.centralWidget()
        if isinstance(central, WelcomeWidget):
            central.set_loading(path_to_workspace)

        self.open_project_action.setEnabled(False)
        self.open_recent_project_submenu.setEnabled(False)

        app = QApplication.instance()
        if app is not None:
            app.processEvents()

        from caliscope.workspace_coordinator import WorkspaceCoordinator

        logger.info(f"Launching session with config file stored in {path_to_workspace}")
        try:
            self.coordinator = WorkspaceCoordinator(Path(path_to_workspace))
            logger.info("Initiate coordinator loading")
            handle = self.coordinator.load_workspace()
        except Exception as e:
            self._on_load_failed(type(e).__name__, str(e))
            return None

        handle.completed.connect(lambda _result: self.build_central_tabs())
        handle.failed.connect(self._on_load_failed)
        self.coordinator.start_load(handle)
        return handle

    def _on_load_failed(self, exc_type: str, message: str) -> None:
        logger.error(f"Workspace load failed: {exc_type}: {message}")
        central = self.centralWidget()
        if isinstance(central, WelcomeWidget):
            central.set_error(message)
        self.open_project_action.setEnabled(True)
        self.open_recent_project_submenu.setEnabled(True)

    def find_tab_index_by_title(self, title):
        # Iterate through tabs to find the index of the tab with the given title
        for index in range(self.central_tab.count()):
            if self.central_tab.tabText(index) == title:
                return index
        return -1  # Return -1 if the tab is not found

    def _navigate_to_tab(self, tab_name: str) -> None:
        """Navigate to requested tab by name.

        Called when "Go to Tab" buttons are clicked in the Project tab.
        Only navigates if the target tab is enabled.
        """
        for i in range(self.central_tab.count()):
            if self.central_tab.tabText(i) == tab_name:
                if self.central_tab.isTabEnabled(i):
                    self.central_tab.setCurrentIndex(i)
                break

    def _on_tab_changed(self, new_index: int) -> None:
        """Suspend/resume 3D rendering when switching tabs.

        QTabWidget doesn't fire hideEvent/showEvent on tab contents when switching
        tabs - it only stops painting them. We manually notify 3D rendering widgets
        when their tab becomes inactive to reduce CPU usage.
        """
        from caliscope.gui.extrinsic_calibration_tab import ExtrinsicCalibrationTab
        from caliscope.gui.reconstruction_tab import ReconstructionTab

        _3d_tab_types = (ExtrinsicCalibrationTab, ReconstructionTab)

        # Suspend rendering on previous tab if it supports it
        prev_widget = self.central_tab.widget(self._previous_tab_index)
        if isinstance(prev_widget, _3d_tab_types):
            logger.debug(f"Suspending rendering on tab {self._previous_tab_index}")
            prev_widget.suspend_rendering()

        # Resume rendering on new tab if it supports it
        new_widget = self.central_tab.widget(new_index)
        if isinstance(new_widget, _3d_tab_types):
            logger.debug(f"Resuming rendering on tab {new_index}")
            new_widget.resume_rendering()

        self._previous_tab_index = new_index

    def add_to_recent_project(self, project_path: str):
        recent_project_action = QAction(project_path, self)
        recent_project_action.triggered.connect(self.open_recent_project)
        self.open_recent_project_submenu.addAction(recent_project_action)

    def open_recent_project(self):
        action = self.sender()
        assert isinstance(action, QAction)
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


def launch_main(workspace: str | None = None):
    from caliscope.gui.gc_confinement import disable, enable

    app = QApplication(sys.argv)
    gc_timer = enable()  # after QApplication, before any Qt3D widgets
    window = MainWindow()
    window.show()
    if workspace is not None:
        window.launch_workspace(workspace)
    app.exec()
    disable(gc_timer)  # after event loop exits, restore automatic GC


if __name__ == "__main__":
    launch_main()
