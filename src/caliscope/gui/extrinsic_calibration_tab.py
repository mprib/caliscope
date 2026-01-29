"""Extrinsic Calibration tab for bundle adjustment workflow.

Glue layer that connects ExtrinsicCalibrationPresenter to the View and Coordinator.
Handles presenter lifecycle and signal wiring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QVBoxLayout, QWidget

from caliscope.gui.presenters.extrinsic_calibration_presenter import (
    ExtrinsicCalibrationPresenter,
)
from caliscope.gui.views.extrinsic_calibration_view import ExtrinsicCalibrationView

if TYPE_CHECKING:
    from caliscope.core.point_data_bundle import PointDataBundle
    from caliscope.workspace_coordinator import WorkspaceCoordinator

logger = logging.getLogger(__name__)


class ExtrinsicCalibrationTab(QWidget):
    """Tab container for extrinsic calibration workflow.

    Creates and manages the presenter/view pair for bundle adjustment
    calibration. On completion, persists the calibrated PointDataBundle
    via the coordinator.

    Lifecycle:
    - Presenter created on tab construction
    - On calibration complete: bundle persisted via coordinator.update_bundle()
    - cleanup() must be called before tab is destroyed
    """

    def __init__(self, coordinator: WorkspaceCoordinator) -> None:
        super().__init__()
        self._coordinator = coordinator
        self._presenter: ExtrinsicCalibrationPresenter | None = None
        self._view: ExtrinsicCalibrationView | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the UI and create presenter/view pair."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create presenter via coordinator factory
        self._presenter = self._coordinator.create_extrinsic_calibration_presenter()

        # Create view with presenter
        self._view = ExtrinsicCalibrationView(self._presenter)
        layout.addWidget(self._view)

        # Wire presenter signals to coordinator
        self._connect_signals()

        logger.info("ExtrinsicCalibrationTab created")

    def _connect_signals(self) -> None:
        """Wire presenter signals to coordinator persistence."""
        if self._presenter is None:
            return

        # Calibration completion - persist bundle via coordinator
        self._presenter.calibration_complete.connect(self._on_calibration_complete)

        # Charuco changes invalidate the presenter - need to recreate
        self._coordinator.charuco_changed.connect(self._on_charuco_changed)

    def _on_calibration_complete(self, bundle: PointDataBundle) -> None:
        """Handle calibration completion - persist bundle via coordinator."""
        logger.info(f"Extrinsic calibration complete: RMSE={bundle.reprojection_report.overall_rmse:.3f}px")

        # Persist via coordinator (handles bundle_updated signal, camera array save, etc.)
        self._coordinator.update_bundle(bundle)

    def _on_charuco_changed(self) -> None:
        """Handle charuco board changes - recreate presenter with new charuco.

        The presenter holds a charuco reference from creation time. When charuco
        changes, the old reference is stale - recreate to get the new one.
        """
        logger.info("Charuco changed - recreating extrinsic calibration presenter")

        # Cleanup existing presenter and view
        if self._presenter is not None:
            self._presenter.cleanup()

        if self._view is not None:
            self._view.cleanup()
            layout = self.layout()
            if layout is not None:
                layout.removeWidget(self._view)
            self._view.deleteLater()

        # Recreate with fresh presenter/view
        self._presenter = self._coordinator.create_extrinsic_calibration_presenter()
        self._view = ExtrinsicCalibrationView(self._presenter)
        layout = self.layout()
        if layout is not None:
            layout.addWidget(self._view)

        self._connect_signals()

    # -------------------------------------------------------------------------
    # VTK Lifecycle
    # -------------------------------------------------------------------------

    def suspend_vtk(self) -> None:
        """Pause VTK rendering when tab is not active."""
        if self._view is not None:
            self._view.suspend_vtk()

    def resume_vtk(self) -> None:
        """Resume VTK rendering when tab becomes active."""
        if self._view is not None:
            self._view.resume_vtk()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def cleanup(self) -> None:
        """Clean up presenter and view resources.

        Must be called before tab is destroyed. The parent (MainWidget) is
        responsible for calling this during reload_workspace or closeEvent.
        """
        if self._view is not None:
            self._view.cleanup()
            self._view = None

        if self._presenter is not None:
            logger.info("Cleaning up extrinsic calibration presenter")
            self._presenter.cleanup()
            self._presenter = None

    def closeEvent(self, event) -> None:
        """Defensive cleanup on normal close."""
        self.cleanup()
        super().closeEvent(event)
