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

        # Charuco changes invalidate the presenter - need to recreate
        self._coordinator.charuco_changed.connect(self._on_charuco_changed)

    def _on_charuco_changed(self) -> None:
        """Update charuco reference when config changes.

        Instead of destroying/recreating the presenter (expensive, causes GUI freeze),
        we hot-swap the charuco reference. Only affects next align_to_origin() call.
        """
        if self._presenter is None:
            return

        self._presenter.update_charuco(self._coordinator.charuco)
        logger.info("Updated charuco in extrinsic calibration presenter")

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
