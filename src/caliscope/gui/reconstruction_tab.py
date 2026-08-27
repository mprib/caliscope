"""Container that wraps ReconstructionWidget with its presenter.

Glue layer: the Tab creates its presenter through the coordinator factory,
forwards coordinator signals to the presenter, and owns their lifecycle.

Call site responsibility (e.g., MainWidget):
    tab = ReconstructionTab(coordinator)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QVBoxLayout, QWidget

from caliscope.gui.views.reconstruction_widget import ReconstructionWidget

if TYPE_CHECKING:
    from caliscope.workspace_coordinator import WorkspaceCoordinator

logger = logging.getLogger(__name__)


class ReconstructionTab(QWidget):
    """Container for ReconstructionWidget.

    Tab owns presenter lifecycle - cleanup must be called before destruction.
    """

    def __init__(
        self,
        coordinator: WorkspaceCoordinator,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._coordinator = coordinator
        self._cleaned_up = False
        self._presenter = coordinator.create_reconstruction_presenter()
        self._widget = ReconstructionWidget(self._presenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._widget)

        self._connect_signals()

        logger.info("ReconstructionTab created")

    def _connect_signals(self) -> None:
        """Wire coordinator signals to this tab."""
        self._coordinator.status_changed.connect(self._refresh_from_workspace)
        self._coordinator.capture_volume_updated.connect(self._on_capture_volume_updated)

    def _refresh_from_workspace(self) -> None:
        """Follow the workspace: recording folders and their assessments."""
        self._presenter.refresh_from_workspace()

    def _on_capture_volume_updated(self) -> None:
        """Rebuild visualization against the recalibrated camera array."""
        self._presenter.refresh_camera_array(self._coordinator.camera_array)

    def cleanup(self) -> None:
        """Clean up resources - call before destruction."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        try:
            self._coordinator.status_changed.disconnect(self._refresh_from_workspace)
        except (RuntimeError, TypeError):
            pass
        try:
            self._coordinator.capture_volume_updated.disconnect(self._on_capture_volume_updated)
        except (RuntimeError, TypeError):
            pass
        self._widget.cleanup()
        self._presenter.cleanup()
        logger.info("ReconstructionTab cleaned up")

    def suspend_rendering(self) -> None:
        """Pause 3D rendering when tab is not active."""
        self._widget.suspend_rendering()

    def resume_rendering(self) -> None:
        """Resume 3D rendering when tab becomes active."""
        self._widget.resume_rendering()

    def closeEvent(self, event) -> None:
        """Defensive cleanup if explicit cleanup wasn't called."""
        self.cleanup()
        super().closeEvent(event)
