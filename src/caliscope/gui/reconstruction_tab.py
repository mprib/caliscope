"""Container that wraps ReconstructionWidget with its presenter.

Thin pass-through layer following the Pure DI pattern: Tab receives its
presenter from the call site rather than creating it internally.

Call site responsibility (e.g., MainWidget):
    presenter = coordinator.create_reconstruction_presenter()
    tab = ReconstructionTab(presenter, coordinator)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QVBoxLayout, QWidget

from caliscope.gui.presenters.reconstruction_presenter import ReconstructionPresenter
from caliscope.gui.views.reconstruction_widget import ReconstructionWidget

if TYPE_CHECKING:
    from caliscope.workspace_coordinator import WorkspaceCoordinator

logger = logging.getLogger(__name__)


class ReconstructionTab(QWidget):
    """Container for ReconstructionWidget.

    Receives presenter via constructor (Pure DI pattern).
    Tab owns presenter lifecycle - cleanup must be called before destruction.
    """

    def __init__(
        self,
        presenter: ReconstructionPresenter,
        coordinator: WorkspaceCoordinator,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._presenter = presenter
        self._coordinator = coordinator
        self._cleaned_up = False
        self._widget = ReconstructionWidget(presenter)
        self._coordinator.status_changed.connect(self._presenter.refresh_recordings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._widget)

        logger.info("ReconstructionTab created")

    def cleanup(self) -> None:
        """Clean up resources - call before destruction."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        try:
            self._coordinator.status_changed.disconnect(self._presenter.refresh_recordings)
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
