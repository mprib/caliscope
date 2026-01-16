"""Cameras tab container widget for intrinsic calibration workflow.

Provides camera selection sidebar and hosts the IntrinsicCalibrationWidget
for the selected camera. Manages presenter lifecycle and forwards calibration
results to the coordinator for persistence.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from caliscope.cameras.camera_array import CameraData
from caliscope.gui.camera_list_widget import CameraListWidget
from caliscope.gui.views.intrinsic_calibration_widget import IntrinsicCalibrationWidget

if TYPE_CHECKING:
    from caliscope.gui.presenters.intrinsic_calibration_presenter import (
        IntrinsicCalibrationPresenter,
    )
    from caliscope.workspace_coordinator import WorkspaceCoordinator

logger = logging.getLogger(__name__)


class CamerasTabWidget(QWidget):
    """Container for Cameras tab with camera list and calibration workflow.

    Layout:
    ┌───────────────────────────────────────────────────────┐
    │ CamerasTabWidget                                      │
    ├──────────────┬────────────────────────────────────────┤
    │              │                                        │
    │ CameraList   │  IntrinsicCalibrationWidget            │
    │ Widget       │  (or message if no video/no selection) │
    │              │                                        │
    └──────────────┴────────────────────────────────────────┘

    Lifecycle:
    - Presenter created lazily on camera selection
    - Presenter cleaned up before switching cameras or closing tab
    - Widget owns presenter cleanup; widget's closeEvent handles render thread
    """

    def __init__(self, coordinator: WorkspaceCoordinator):
        super().__init__()
        self.coordinator = coordinator

        # Current presenter/widget (one at a time)
        self._current_presenter: IntrinsicCalibrationPresenter | None = None
        self._current_widget: IntrinsicCalibrationWidget | None = None

        self._setup_ui()
        self._connect_signals()

        # Auto-select first camera if available
        if self.camera_list.count() > 0:
            self.camera_list.setCurrentRow(0)

    def _setup_ui(self) -> None:
        """Build the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Use splitter for resizable sidebar
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Camera list
        self.camera_list = CameraListWidget(self.coordinator.camera_array)
        self._splitter.addWidget(self.camera_list)

        # Right: Content area (placeholder initially)
        self._content_container = QWidget()
        self._content_layout = QVBoxLayout(self._content_container)
        self._content_layout.setContentsMargins(0, 0, 0, 0)

        self._message_label = QLabel("Select a camera to begin calibration")
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message_label.setStyleSheet("color: #888; font-size: 14px;")
        self._content_layout.addWidget(self._message_label)

        self._splitter.addWidget(self._content_container)

        # Set initial sizes (sidebar narrower than content)
        self._splitter.setSizes([200, 800])

        layout.addWidget(self._splitter)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self.camera_list.camera_selected.connect(self._on_camera_selected)

    def _on_camera_selected(self, port: int) -> None:
        """Handle camera selection - create presenter and widget."""
        logger.info(f"Camera selected: port {port}")

        # Clean up previous presenter/widget
        self._cleanup_current()

        try:
            presenter = self.coordinator.create_intrinsic_presenter(port)
        except ValueError as e:
            # Video missing or camera not in array
            logger.warning(f"Cannot create presenter for port {port}: {e}")
            self._show_message(str(e))
            return

        # Store reference for cleanup
        self._current_presenter = presenter

        # Connect calibration_complete to coordinator persistence
        presenter.calibration_complete.connect(partial(self._on_calibration_complete, port))

        # Create and show the calibration widget
        widget = IntrinsicCalibrationWidget(presenter)
        self._current_widget = widget

        # Replace message with widget
        self._message_label.hide()
        self._content_layout.addWidget(widget)

        logger.info(f"Intrinsic calibration widget active for port {port}")

    def _on_calibration_complete(self, port: int, camera: CameraData) -> None:
        """Handle calibration completion - persist and update list."""
        logger.info(f"Calibration complete for port {port}, RMSE: {camera.error}")

        # Persist to ground truth via coordinator
        self.coordinator.persist_intrinsic_calibration(camera)

        # Refresh camera list to show updated status
        self.camera_list.refresh(self.coordinator.camera_array)

    def _show_message(self, text: str) -> None:
        """Show a message in the content area."""
        # Remove current widget if any
        if self._current_widget is not None:
            self._content_layout.removeWidget(self._current_widget)
            self._current_widget.close()
            self._current_widget.deleteLater()
            self._current_widget = None

        self._message_label.setText(text)
        self._message_label.show()

    def _cleanup_current(self) -> None:
        """Clean up current presenter and widget."""
        if self._current_presenter is not None:
            logger.info("Cleaning up current presenter")
            self._current_presenter.cleanup()
            self._current_presenter = None

        if self._current_widget is not None:
            logger.info("Cleaning up current widget")
            self._content_layout.removeWidget(self._current_widget)
            self._current_widget.close()  # Triggers closeEvent for render thread
            self._current_widget.deleteLater()
            self._current_widget = None

    def cleanup(self) -> None:
        """Explicit cleanup - MUST be called before destruction.

        Note: closeEvent is NOT reliable for tab widgets because
        removeTab() + deleteLater() doesn't trigger closeEvent.
        The parent (MainWidget) must call this during reload_workspace.
        """
        self._cleanup_current()

    def closeEvent(self, event) -> None:
        """Defensive cleanup on normal close."""
        self.cleanup()
        super().closeEvent(event)
