"""Multi-Camera Processing tab for synchronized video 2D extraction.

Glue layer that connects MultiCameraProcessingPresenter to the View and Coordinator.
Handles presenter lifecycle and signal wiring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from caliscope.gui.presenters.multi_camera_processing_presenter import (
    MultiCameraProcessingPresenter,
)
from caliscope.gui.theme import Colors
from caliscope.gui.views.multi_camera_processing_widget import MultiCameraProcessingWidget

if TYPE_CHECKING:
    from caliscope.workspace_coordinator import WorkspaceCoordinator

logger = logging.getLogger(__name__)


class MultiCameraProcessingTab(QWidget):
    """Tab container for multi-camera synchronized video processing.

    Creates and manages the presenter/view pair for extracting 2D landmarks
    from synchronized multi-camera video. On completion, persists ImagePoints
    via the coordinator.

    Lifecycle:
    - Presenter created on tab construction
    - Presenter configured with extrinsic recording dir and cameras
    - On processing complete: ImagePoints persisted, next tab enabled
    - cleanup() must be called before tab is destroyed
    """

    def __init__(self, coordinator: WorkspaceCoordinator) -> None:
        super().__init__()
        self.coordinator = coordinator
        self._presenter: MultiCameraProcessingPresenter | None = None
        self._widget: MultiCameraProcessingWidget | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the UI and create presenter/widget pair."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._file_warning_label = QLabel()
        self._file_warning_label.setTextFormat(Qt.TextFormat.PlainText)
        self._file_warning_label.setWordWrap(True)
        self._file_warning_label.setStyleSheet(f"color: {Colors.WARNING};")
        self._file_warning_label.setContentsMargins(8, 8, 8, 0)
        self._file_warning_label.hide()
        layout.addWidget(self._file_warning_label)

        # Create presenter via coordinator factory
        self._presenter = self.coordinator.create_multi_camera_presenter()

        # Configure presenter with extrinsic recording directory and cameras
        self._presenter.set_recording_dir(self.coordinator.workspace_guide.extrinsic_dir)
        self._presenter.set_cameras(self.coordinator.camera_array.cameras)

        # Create view
        self._widget = MultiCameraProcessingWidget(self._presenter)
        layout.addWidget(self._widget)

        # Wire presenter signals to coordinator
        self._connect_signals()
        self._refresh_from_workspace()

    def _connect_signals(self) -> None:
        """Wire presenter signals to coordinator persistence."""
        if self._presenter is None:
            return

        # Rotation persistence
        self._presenter.rotation_changed.connect(self.coordinator.persist_camera_rotation)

        # Target changes invalidate the tracker - need to hot-swap it
        self.coordinator.extrinsic_target_changed.connect(self._on_extrinsic_target_changed)

        # Cameras and videos can appear or disappear after this tab is built.
        self.coordinator.status_changed.connect(self._refresh_from_workspace)

    def _refresh_from_workspace(self) -> None:
        """Follow the workspace: camera set, video files, and file feedback."""
        if self._presenter is not None:
            cameras = self.coordinator.camera_array.cameras
            if set(cameras) != set(self._presenter.cameras):
                self._presenter.set_cameras(cameras)
            else:
                self._presenter.refresh_videos()

        issues = self.coordinator.get_workflow_status().extrinsic_video_issues
        if issues:
            self._file_warning_label.setText(
                "Extrinsic videos need attention:\n" + "\n".join(issue.message for issue in issues)
            )
            self._file_warning_label.show()
        else:
            self._file_warning_label.clear()
            self._file_warning_label.hide()

    def _on_extrinsic_target_changed(self) -> None:
        """Update tracker when extrinsic target config changes.

        Instead of destroying/recreating the presenter (expensive, causes GUI freeze),
        we hot-swap the tracker reference. Results are cleared since point IDs change.
        """
        if self._presenter is None:
            return

        new_tracker = self.coordinator.create_extrinsic_tracker()
        self._presenter.update_tracker(new_tracker)
        logger.info("Updated tracker in multi-camera presenter")

    def cleanup(self) -> None:
        """Clean up presenter resources.

        Must be called before tab is destroyed. The parent (MainWidget) is
        responsible for calling this during reload_workspace or closeEvent.
        """
        if self._presenter is not None:
            # The coordinator outlives this tab; drop the status_changed
            # connection so it never fires into a deleted label.
            try:
                self.coordinator.status_changed.disconnect(self._refresh_from_workspace)
            except (RuntimeError, TypeError):
                pass
            logger.info("Cleaning up multi-camera processing presenter")
            self._presenter.cleanup()
            self._presenter = None

    def closeEvent(self, event) -> None:
        """Defensive cleanup on normal close."""
        self.cleanup()
        super().closeEvent(event)
