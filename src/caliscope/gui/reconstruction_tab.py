"""Container that wraps ReconstructionWidget with its presenter.

Thin pass-through layer following the Pure DI pattern: Tab receives its
presenter from the call site rather than creating it internally.

Call site responsibility (e.g., MainWidget):
    presenter = coordinator.create_reconstruction_presenter()
    tab = ReconstructionTab(presenter)
"""

from __future__ import annotations

import logging

from PySide6.QtWidgets import QVBoxLayout, QWidget

from caliscope.gui.presenters.reconstruction_presenter import ReconstructionPresenter
from caliscope.gui.views.reconstruction_widget import ReconstructionWidget

logger = logging.getLogger(__name__)


class ReconstructionTab(QWidget):
    """Container for ReconstructionWidget.

    Receives presenter via constructor (Pure DI pattern).
    Tab owns presenter lifecycle - cleanup must be called before destruction.
    """

    def __init__(
        self,
        presenter: ReconstructionPresenter,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._presenter = presenter
        self._widget = ReconstructionWidget(presenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._widget)

        logger.info("ReconstructionTab created")

    def cleanup(self) -> None:
        """Clean up resources - call before destruction."""
        self._widget.cleanup()
        self._presenter.cleanup()
        logger.info("ReconstructionTab cleaned up")

    def suspend_vtk(self) -> None:
        """Pause VTK rendering when tab is not active."""
        self._widget.suspend_vtk()

    def resume_vtk(self) -> None:
        """Resume VTK rendering when tab becomes active."""
        self._widget.resume_vtk()

    def closeEvent(self, event) -> None:
        """Defensive cleanup if explicit cleanup wasn't called."""
        self.cleanup()
        super().closeEvent(event)
