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
from caliscope.gui.tab_names import TabName
from caliscope.gui.widgets.welcome_widget import WelcomeWidget

if TYPE_CHECKING:
    from caliscope.task_manager import TaskHandle

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.app_settings = rtoml.load(APP_SETTINGS_PATH)

        # Bumped on every launch_workspace so a relaunch invalidates any pending
        # deferred tab build from a previous project (see _build_next_deferred_tab).
        self._build_generation: int = 0

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

        # Invalidate any pending deferred tab tick or queued completed signal so
        # neither builds tabs against the coordinator cleaned up below.
        self._build_generation += 1

        self._cleanup_tabs()

        # Coordinator cleanup (TaskManager shutdown)
        if hasattr(self, "coordinator"):
            self.coordinator.cleanup()

        logger.info("Application cleanup complete")
        super().closeEvent(event)

    def _cleanup_tabs(self) -> None:
        """Release per-tab resources (background threads, Qt3D scene graphs).

        Shared by closeEvent and launch_workspace so the two teardown paths
        can't drift. Safe to call before any tabs exist.
        """
        from caliscope.gui.cameras_tab_widget import CamerasTabWidget
        from caliscope.gui.extrinsic_calibration_tab import ExtrinsicCalibrationTab
        from caliscope.gui.multi_camera_processing_tab import MultiCameraProcessingTab
        from caliscope.gui.reconstruction_tab import ReconstructionTab

        cameras = getattr(self, "cameras_tab_widget", None)
        if isinstance(cameras, CamerasTabWidget):
            cameras.cleanup()
        multi = getattr(self, "multi_camera_tab", None)
        if isinstance(multi, MultiCameraProcessingTab):
            multi.cleanup()
        extrinsic = getattr(self, "extrinsic_calibration_tab", None)
        if isinstance(extrinsic, ExtrinsicCalibrationTab):
            extrinsic.cleanup()
        recon = getattr(self, "reconstruction_tab", None)
        if isinstance(recon, ReconstructionTab):
            recon.cleanup()

        # Drop the attributes so a second call (project switch, then app close)
        # never touches widgets whose C++ side setCentralWidget already deleted.
        for name in ("cameras_tab_widget", "multi_camera_tab", "extrinsic_calibration_tab", "reconstruction_tab"):
            if hasattr(self, name):
                delattr(self, name)

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

    def build_central_tabs(self, gen: int) -> None:
        # Same invalidation contract as _build_next_deferred_tab: a relaunch or
        # app close bumped the generation, so a queued completed signal from a
        # torn-down coordinator must not build tabs against it.
        if gen != self._build_generation:
            return
        self.central_tab = QTabWidget(self)
        self.setCentralWidget(self.central_tab)

        self.statusBar().showMessage("Building Project tab…")

        from caliscope.gui.views.project_setup_view import ProjectSetupView

        logger.info("Building Project setup tab")
        self.project_tab = ProjectSetupView(self.coordinator)
        self.project_tab.tab_navigation_requested.connect(self._navigate_to_tab)
        self.central_tab.addTab(self.project_tab, TabName.PROJECT)

        self.coordinator.status_changed.connect(self._refresh_tab_enablement)
        self._previous_tab_index: int = 0
        self.central_tab.currentChanged.connect(self._on_tab_changed)

        # Remaining tabs added one per event loop tick so the UI stays responsive.
        # Capture the current build generation so a relaunch mid-build abandons this
        # chain instead of touching a torn-down coordinator (see _build_next_deferred_tab).
        gen = self._build_generation
        self._deferred_tab_builders: deque[tuple[str, Callable[[], None]]] = deque(
            [
                (TabName.CAMERAS, self._add_cameras_tab),
                (TabName.MULTI_CAMERA, self._add_multi_camera_tab),
                (TabName.CALIBRATE, self._add_calibrate_tab),
                (TabName.RECONSTRUCTION, self._add_reconstruction_tab),
            ]
        )
        QTimer.singleShot(0, lambda: self._build_next_deferred_tab(gen))

    def _build_next_deferred_tab(self, gen: int) -> None:
        # A relaunch bumped the generation; abandon this stale build chain.
        if gen != self._build_generation:
            return
        if not self._deferred_tab_builders:
            self._on_deferred_build_complete()
            return
        label, builder = self._deferred_tab_builders.popleft()
        self.statusBar().showMessage(f"Building {label} tab…")
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        builder()
        if self._deferred_tab_builders:
            QTimer.singleShot(0, lambda: self._build_next_deferred_tab(gen))
        else:
            self._on_deferred_build_complete()

    def _on_deferred_build_complete(self) -> None:
        """Deferred tab chain finished: restore Open Project access, clear status.

        Open Project stays disabled through the whole build so a second launch
        can't race a half-built tab set.
        """
        self.open_project_action.setEnabled(True)
        self.open_recent_project_submenu.setEnabled(True)
        self.statusBar().showMessage("Ready", 3000)

    # Each _make_*_tab constructs a tab widget (real or placeholder) and reports
    # whether it should be enabled. The _add_* builders append at the end during the
    # initial deferred build; _replace_placeholder_tab swaps one in at its existing
    # index once it becomes enabled. Construction lives here so the two paths share it.

    def _make_cameras_tab(self) -> tuple[QWidget, bool]:
        from caliscope.gui.cameras_tab_widget import CamerasTabWidget
        from caliscope.gui.widgets.cameras_info_placeholder import CamerasInfoPlaceholder

        if self.coordinator.cameras_tab_enabled:
            logger.info("Building Cameras tab with intrinsic calibration")
            self.cameras_tab_widget: QWidget = CamerasTabWidget(self.coordinator)
        else:
            logger.info("No intrinsic videos - Cameras tab shows skip-intrinsics placeholder")
            self.cameras_tab_widget = CamerasInfoPlaceholder()
        # Cameras stays interactive even as a placeholder (it explains skip-intrinsics).
        return self.cameras_tab_widget, True

    def _make_multi_camera_tab(self) -> tuple[QWidget, bool]:
        from caliscope.gui.multi_camera_processing_tab import MultiCameraProcessingTab

        enabled = self.coordinator.multi_camera_tab_enabled
        if enabled:
            logger.info("Building Multi-Camera processing tab")
            self.multi_camera_tab: QWidget = MultiCameraProcessingTab(self.coordinator)
        else:
            self.multi_camera_tab = QWidget()
        return self.multi_camera_tab, enabled

    def _make_calibrate_tab(self) -> tuple[QWidget, bool]:
        from caliscope.gui.extrinsic_calibration_tab import ExtrinsicCalibrationTab

        enabled = self.coordinator.capture_volume_tab_enabled
        if enabled:
            logger.info("Creating ExtrinsicCalibrationTab")
            self.extrinsic_calibration_tab: QWidget = ExtrinsicCalibrationTab(self.coordinator)
            self.extrinsic_calibration_tab.navigation_requested.connect(self._navigate_to_tab)  # type: ignore[union-attr]
        else:
            self.extrinsic_calibration_tab = QWidget()
        return self.extrinsic_calibration_tab, enabled

    def _make_reconstruction_tab(self) -> tuple[QWidget, bool]:
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
        return self.reconstruction_tab, enabled

    def _add_cameras_tab(self) -> None:
        widget, _enabled = self._make_cameras_tab()
        self.central_tab.addTab(widget, TabName.CAMERAS)

    def _add_multi_camera_tab(self) -> None:
        widget, enabled = self._make_multi_camera_tab()
        self.central_tab.addTab(widget, TabName.MULTI_CAMERA)
        self.central_tab.setTabEnabled(self.find_tab_index_by_title(TabName.MULTI_CAMERA), enabled)

    def _add_calibrate_tab(self) -> None:
        widget, enabled = self._make_calibrate_tab()
        self.central_tab.addTab(widget, TabName.CALIBRATE)
        self.central_tab.setTabEnabled(self.find_tab_index_by_title(TabName.CALIBRATE), enabled)

    def _add_reconstruction_tab(self) -> None:
        widget, enabled = self._make_reconstruction_tab()
        self.central_tab.addTab(widget, TabName.RECONSTRUCTION)
        self.central_tab.setTabEnabled(self.find_tab_index_by_title(TabName.RECONSTRUCTION), enabled)

    def _replace_placeholder_tab(self, tab_name: TabName) -> None:
        from caliscope.gui.cameras_tab_widget import CamerasTabWidget
        from caliscope.gui.extrinsic_calibration_tab import ExtrinsicCalibrationTab
        from caliscope.gui.multi_camera_processing_tab import MultiCameraProcessingTab
        from caliscope.gui.reconstruction_tab import ReconstructionTab

        real_tab_types: dict[TabName, type[QWidget]] = {
            TabName.CAMERAS: CamerasTabWidget,
            TabName.MULTI_CAMERA: MultiCameraProcessingTab,
            TabName.CALIBRATE: ExtrinsicCalibrationTab,
            TabName.RECONSTRUCTION: ReconstructionTab,
        }
        makers: dict[TabName, Callable[[], tuple[QWidget, bool]]] = {
            TabName.CAMERAS: self._make_cameras_tab,
            TabName.MULTI_CAMERA: self._make_multi_camera_tab,
            TabName.CALIBRATE: self._make_calibrate_tab,
            TabName.RECONSTRUCTION: self._make_reconstruction_tab,
        }

        idx = self.find_tab_index_by_title(tab_name)
        if idx < 0:
            return
        old = self.central_tab.widget(idx)
        if isinstance(old, real_tab_types[tab_name]):
            return

        widget, enabled = makers[tab_name]()
        self.central_tab.removeTab(idx)
        self.central_tab.insertTab(idx, widget, tab_name)
        self.central_tab.setTabEnabled(idx, enabled)

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
            self.find_tab_index_by_title(TabName.MULTI_CAMERA),
            self.coordinator.multi_camera_tab_enabled,
        )
        self.central_tab.setTabEnabled(
            self.find_tab_index_by_title(TabName.CALIBRATE),
            self.coordinator.capture_volume_tab_enabled,
        )
        self.central_tab.setTabEnabled(
            self.find_tab_index_by_title(TabName.RECONSTRUCTION),
            self.coordinator.reconstruction_tab_enabled,
        )

        # If a tab became enabled and has a dummy widget, replace it
        self._maybe_replace_dummy_tabs()

    def _maybe_replace_dummy_tabs(self) -> None:
        """Replace dummy widgets with real tabs when they become enabled."""
        if self.coordinator.cameras_tab_enabled:
            self._replace_placeholder_tab(TabName.CAMERAS)
        if self.coordinator.multi_camera_tab_enabled:
            self._replace_placeholder_tab(TabName.MULTI_CAMERA)
        if self.coordinator.capture_volume_tab_enabled:
            self._replace_placeholder_tab(TabName.CALIBRATE)
        if self.coordinator.reconstruction_tab_enabled:
            self._replace_placeholder_tab(TabName.RECONSTRUCTION)

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
        # Invalidate any in-flight deferred tab build from a previous project so its
        # timer chain can't touch the coordinator we are about to tear down.
        self._build_generation += 1

        # If a project is already open, release its tabs' resources while the
        # coordinator is still alive (same order as closeEvent), then swap in a
        # fresh welcome screen — setCentralWidget deletes the old central widget,
        # and Qt3D scene graphs must be torn down before that. The swap gives
        # set_loading/_on_load_failed a valid target on every load path.
        central = self.centralWidget()
        if not isinstance(central, WelcomeWidget):
            self._cleanup_tabs()
            central = WelcomeWidget(self.recent_projects())
            central.open_project_requested.connect(self.create_new_project_folder)
            central.recent_project_selected.connect(self.launch_workspace)
            self.setCentralWidget(central)

        # Tear down existing coordinator to avoid orphaned thread pools
        if hasattr(self, "coordinator"):
            self.coordinator.cleanup()
            del self.coordinator

        # Show loading state on welcome widget and flush the paint
        # before the blocking WorkspaceCoordinator constructor
        central.set_loading(path_to_workspace)

        self.open_project_action.setEnabled(False)
        self.open_recent_project_submenu.setEnabled(False)

        app = QApplication.instance()
        if app is not None:
            app.processEvents()

        from caliscope.workspace_coordinator import WorkspaceCoordinator

        logger.info(f"Launching session with config file stored in {path_to_workspace}")
        # Leak-safety invariant: a settings error raises inside WorkspaceCoordinator.__init__
        # before its TaskManager is constructed, so the synchronous failure path below leaks
        # nothing (no threads exist yet to orphan).
        try:
            self.coordinator = WorkspaceCoordinator(Path(path_to_workspace))
            logger.info("Initiate coordinator loading")
            handle = self.coordinator.load_workspace()
        except Exception as e:
            self._on_load_failed(type(e).__name__, str(e))
            return None

        handle.completed.connect(lambda _result, gen=self._build_generation: self.build_central_tabs(gen))
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

    def find_tab_index_by_title(self, title: str) -> int:
        for index in range(self.central_tab.count()):
            if self.central_tab.tabText(index) == title:
                return index
        # A miss feeds -1 into Qt calls that silently no-op (setTabEnabled,
        # setCurrentIndex), so this warning is the only trace of a stale title.
        logger.warning(f"Tab lookup missed: no tab titled {title!r}")
        return -1

    def _navigate_to_tab(self, tab_name: str) -> None:
        """Navigate to requested tab by name.

        Called when "Go to Tab" buttons are clicked in the Project tab.
        Only navigates if the target tab is enabled.
        """
        idx = self.find_tab_index_by_title(tab_name)
        if idx >= 0 and self.central_tab.isTabEnabled(idx):
            self.central_tab.setCurrentIndex(idx)

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
