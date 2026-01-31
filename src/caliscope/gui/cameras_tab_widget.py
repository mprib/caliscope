"""Cameras tab container widget for intrinsic calibration workflow.

Provides camera selection sidebar and hosts the IntrinsicCalibrationWidget
for the selected camera. Uses pool pattern — presenters are kept alive when
switching cameras, allowing calibration to continue in background.
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

from caliscope.core.calibrate_intrinsics import IntrinsicCalibrationOutput
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
    ┌───────────────────────────────────────────────────────────┐
    │ CamerasTabWidget                                          │
    ├──────────────┬────────────────────────────────────────────┤
    │              │                                            │
    │ CameraList   │  IntrinsicCalibrationWidget                │
    │ Widget       │  (or message if no video/no selection)     │
    │              │                                            │
    └──────────────┴────────────────────────────────────────────┘

    Lifecycle:
    - Presenters created lazily on first camera selection
    - Presenters kept alive when switching cameras (pool pattern)
    - All presenters cleaned up when tab is closed or workspace reloaded
    """

    def __init__(self, coordinator: WorkspaceCoordinator):
        super().__init__()
        self.coordinator = coordinator

        # Pool of presenters and widgets, keyed by port
        self._presenters: dict[int, IntrinsicCalibrationPresenter] = {}
        self._widgets: dict[int, IntrinsicCalibrationWidget] = {}
        self._current_port: int | None = None

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
        self.camera_list.setMinimumWidth(150)  # Prevent collapse
        self._splitter.addWidget(self.camera_list)

        # Right: Content area (placeholder initially)
        self._content_container = QWidget()
        self._content_container.setMinimumWidth(400)  # Prevent collapse
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
        self.coordinator.charuco_changed.connect(self._on_charuco_changed)

    def _on_charuco_changed(self) -> None:
        """Update tracker in all pooled presenters when charuco changes.

        Instead of destroying/recreating presenters (expensive, causes GUI freeze),
        we hot-swap the tracker reference in each. The streamer keeps the old
        tracker until calibration restarts, which is acceptable.
        """
        new_tracker = self.coordinator.create_tracker()
        for port, presenter in self._presenters.items():
            presenter.update_tracker(new_tracker)
        logger.info(f"Updated tracker in {len(self._presenters)} pooled presenters")

    def _on_camera_selected(self, port: int) -> None:
        """Handle camera selection - show existing or create new presenter/widget."""
        logger.info(f"Camera selected: port {port}")

        # Hide current widget (keep presenter running in background)
        if self._current_port is not None and self._current_port in self._widgets:
            current_widget = self._widgets[self._current_port]
            self._content_layout.removeWidget(current_widget)
            current_widget.hide()

        # Get or create presenter/widget for new port
        if port not in self._presenters:
            try:
                presenter = self.coordinator.create_intrinsic_presenter(port)
            except ValueError as e:
                logger.warning(f"Cannot create presenter for port {port}: {e}")
                self._show_message(str(e))
                return

            presenter.calibration_complete.connect(partial(self._on_calibration_complete, port))
            widget = IntrinsicCalibrationWidget(presenter)

            self._presenters[port] = presenter
            self._widgets[port] = widget

        # Show the widget for this port
        widget = self._widgets[port]
        self._message_label.hide()
        self._content_layout.addWidget(widget)
        widget.show()
        self._current_port = port

        logger.info(f"Intrinsic calibration widget active for port {port}")

    def _on_calibration_complete(self, port: int, output: IntrinsicCalibrationOutput) -> None:
        """Handle calibration completion - persist and update list."""
        report = output.report
        logger.info(f"Calibration complete for port {port}, rmse={report.rmse:.3f}px")

        # Get collected points from presenter for session-based overlay restoration
        collected_points = None
        if port in self._presenters:
            collected_points = self._presenters[port].collected_points

        # Persist to ground truth via coordinator (including collected points for session)
        self.coordinator.persist_intrinsic_calibration(output, collected_points)

        # Refresh camera list to show updated status
        self.camera_list.refresh(self.coordinator.camera_array)

    def _show_message(self, text: str) -> None:
        """Show a message in the content area."""
        # Hide current widget if any
        if self._current_port is not None and self._current_port in self._widgets:
            current_widget = self._widgets[self._current_port]
            self._content_layout.removeWidget(current_widget)
            current_widget.hide()

        self._message_label.setText(text)
        self._message_label.show()

    def cleanup(self) -> None:
        """Clean up all presenters and widgets.

        Note: closeEvent is NOT reliable for tab widgets because
        removeTab() + deleteLater() doesn't trigger closeEvent.
        The parent (MainWidget) must call this during reload_workspace.
        """
        for port, presenter in self._presenters.items():
            logger.info(f"Cleaning up presenter for port {port}")
            presenter.cleanup()

        for port, widget in self._widgets.items():
            logger.info(f"Cleaning up widget for port {port}")
            self._content_layout.removeWidget(widget)
            widget.close()
            widget.deleteLater()

        self._presenters.clear()
        self._widgets.clear()
        self._current_port = None

    def closeEvent(self, event) -> None:
        """Defensive cleanup on normal close."""
        self.cleanup()
        super().closeEvent(event)
