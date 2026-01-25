"""Multi-Camera Processing tab for synchronized video 2D extraction.

Glue layer that connects MultiCameraProcessingPresenter to the View and Coordinator.
Handles presenter lifecycle and signal wiring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QVBoxLayout, QWidget

from caliscope.gui.presenters.multi_camera_processing_presenter import (
    MultiCameraProcessingPresenter,
)
from caliscope.gui.views.multi_camera_processing_widget import MultiCameraProcessingWidget

if TYPE_CHECKING:
    from caliscope.core.coverage_analysis import ExtrinsicCoverageReport
    from caliscope.core.point_data import ImagePoints
    from caliscope.tracker import Tracker
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

    def _connect_signals(self) -> None:
        """Wire presenter signals to coordinator persistence."""
        if self._presenter is None:
            return

        # Rotation persistence
        self._presenter.rotation_changed.connect(self.coordinator.persist_camera_rotation)

        # Processing completion - persist results and notify coordinator
        self._presenter.processing_complete.connect(self._on_processing_complete)

        # Charuco changes invalidate the tracker - need to recreate presenter
        self.coordinator.charuco_changed.connect(self._on_charuco_changed)

    def _on_processing_complete(
        self,
        image_points: ImagePoints,
        coverage_report: ExtrinsicCoverageReport,
        tracker: Tracker,
    ) -> None:
        """Handle processing completion - persist results and signal coordinator."""
        logger.info(f"Multi-camera processing complete: {len(image_points.df)} observations")

        # Persist ImagePoints to extrinsic directory
        self.coordinator.persist_extrinsic_image_points(image_points, tracker.name)

        # Signal that ImagePoints are ready (enables Tab 2)
        self.coordinator.extrinsic_image_points_ready.emit()

    def _on_charuco_changed(self) -> None:
        """Handle charuco board changes - recreate presenter with new tracker.

        The presenter holds a tracker reference from creation time. When charuco
        changes, the old tracker is stale - recreate to get the new one.
        """
        logger.info("Charuco changed - recreating multi-camera presenter")

        # Cleanup existing presenter
        if self._presenter is not None:
            self._presenter.cleanup()

        # Remove old widget
        if self._widget is not None:
            layout = self.layout()
            if layout is not None:
                layout.removeWidget(self._widget)
            self._widget.deleteLater()

        # Recreate with fresh presenter/widget
        self._presenter = self.coordinator.create_multi_camera_presenter()
        self._presenter.set_recording_dir(self.coordinator.workspace_guide.extrinsic_dir)
        self._presenter.set_cameras(self.coordinator.camera_array.cameras)

        self._widget = MultiCameraProcessingWidget(self._presenter)
        layout = self.layout()
        if layout is not None:
            layout.addWidget(self._widget)

        self._connect_signals()

    def cleanup(self) -> None:
        """Clean up presenter resources.

        Must be called before tab is destroyed. The parent (MainWidget) is
        responsible for calling this during reload_workspace or closeEvent.
        """
        if self._presenter is not None:
            logger.info("Cleaning up multi-camera processing presenter")
            self._presenter.cleanup()
            self._presenter = None

    def closeEvent(self, event) -> None:
        """Defensive cleanup on normal close."""
        self.cleanup()
        super().closeEvent(event)
