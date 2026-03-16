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
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from caliscope.core.calibrate_intrinsics import IntrinsicCalibrationOutput
from caliscope.gui.theme import Colors
from caliscope.gui.utils.chessboard_preview import render_chessboard_pixmap
from caliscope.gui.utils.charuco_preview import render_charuco_pixmap
from caliscope.gui.utils.spinbox_utils import setup_spinbox_sizing
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

        # Pool of presenters and widgets, keyed by cam_id
        self._presenters: dict[int, IntrinsicCalibrationPresenter] = {}
        self._widgets: dict[int, IntrinsicCalibrationWidget] = {}
        self._current_cam_id: int | None = None

        self._setup_ui()
        self._connect_signals()
        self._update_pattern_preview()

        # Auto-select first camera if available
        if self.camera_list.count() > 0:
            self.camera_list.setCurrentRow(0)

    def _setup_ui(self) -> None:
        """Build the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Use splitter for resizable sidebar
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Camera list + pattern preview (vertical stack)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.camera_list = CameraListWidget(self.coordinator.camera_array)
        self.camera_list.setMinimumWidth(150)
        left_layout.addWidget(self.camera_list, stretch=1)

        # Chessboard reference preview (read-only)
        self._pattern_preview = QLabel()
        self._pattern_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pattern_preview.setMaximumHeight(160)
        self._pattern_preview.setStyleSheet(
            f"QLabel {{ background-color: {Colors.SURFACE}; border: 1px solid {Colors.BORDER}; border-radius: 4px; }}"
        )
        left_layout.addWidget(self._pattern_preview)

        self._pattern_info = QLabel()
        self._pattern_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pattern_info.setStyleSheet("color: #888; font-size: 11px;")
        left_layout.addWidget(self._pattern_info)

        # Frame skip control (global for all cameras)
        frame_skip_label = QLabel("Process every")
        frame_skip_label.setStyleSheet("color: #aaa; font-size: 11px;")
        left_layout.addWidget(frame_skip_label)

        self._frame_skip_spin = QSpinBox()
        self._frame_skip_spin.setValue(self.coordinator.intrinsic_frame_skip)
        self._frame_skip_spin.setToolTip(
            "Process every Nth frame during calibration.\n\n"
            "1 = every frame (slow, most data)\n"
            "5 = every 5th frame (good balance)\n"
            "Higher values speed up processing but use fewer frames."
        )
        setup_spinbox_sizing(self._frame_skip_spin, min_value=1, max_value=30)
        left_layout.addWidget(self._frame_skip_spin)

        self._splitter.addWidget(left_container)

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
        self.coordinator.intrinsic_target_changed.connect(self._on_intrinsic_target_changed)
        self._frame_skip_spin.valueChanged.connect(self._on_frame_skip_changed)

    def _on_intrinsic_target_changed(self) -> None:
        """Update tracker in all pooled presenters and refresh preview."""
        new_tracker = self.coordinator.create_intrinsic_tracker()
        for cam_id, presenter in self._presenters.items():
            presenter.update_tracker(new_tracker)
        logger.info(f"Updated tracker in {len(self._presenters)} pooled presenters")
        self._update_pattern_preview()

    def _on_frame_skip_changed(self, value: int) -> None:
        """Propagate frame skip change to coordinator and all active presenters."""
        self.coordinator.set_intrinsic_frame_skip(value, self._presenters)

    def _on_camera_selected(self, cam_id: int) -> None:
        """Handle camera selection - show existing or create new presenter/widget."""
        logger.info(f"Camera selected: cam {cam_id}")

        # Hide current widget (keep presenter running in background)
        if self._current_cam_id is not None and self._current_cam_id in self._widgets:
            current_widget = self._widgets[self._current_cam_id]
            self._content_layout.removeWidget(current_widget)
            current_widget.hide()

        # Get or create presenter/widget for new cam_id
        if cam_id not in self._presenters:
            try:
                presenter = self.coordinator.create_intrinsic_presenter(cam_id)
            except ValueError as e:
                logger.warning(f"Cannot create presenter for cam {cam_id}: {e}")
                self._show_message(str(e))
                return

            presenter.calibration_complete.connect(partial(self._on_calibration_complete, cam_id))
            widget = IntrinsicCalibrationWidget(presenter)

            self._presenters[cam_id] = presenter
            self._widgets[cam_id] = widget

        # Show the widget for this cam_id
        widget = self._widgets[cam_id]
        self._message_label.hide()
        self._content_layout.addWidget(widget)
        widget.show()
        self._current_cam_id = cam_id

        logger.info(f"Intrinsic calibration widget active for cam {cam_id}")

    def _on_calibration_complete(self, cam_id: int, output: IntrinsicCalibrationOutput) -> None:
        """Handle calibration completion - persist and update list."""
        report = output.report
        logger.info(f"Calibration complete for cam {cam_id}, rmse={report.rmse:.3f}px")

        # Get collected points from presenter for session-based overlay restoration
        collected_points = None
        if cam_id in self._presenters:
            collected_points = self._presenters[cam_id].collected_points

        # Persist to ground truth via coordinator (including collected points for session)
        self.coordinator.persist_intrinsic_calibration(output, collected_points)

        # Refresh camera list to show updated status
        self.camera_list.refresh(self.coordinator.camera_array)

    def _show_message(self, text: str) -> None:
        """Show a message in the content area."""
        # Hide current widget if any
        if self._current_cam_id is not None and self._current_cam_id in self._widgets:
            current_widget = self._widgets[self._current_cam_id]
            self._content_layout.removeWidget(current_widget)
            current_widget.hide()

        self._message_label.setText(text)
        self._message_label.show()

    def _update_pattern_preview(self) -> None:
        """Update the pattern preview from coordinator state.

        Queries the repository for the current intrinsic target type,
        then renders the appropriate preview.
        """
        target_type = self.coordinator.targets_repository.intrinsic_target_type

        if target_type == "chessboard":
            if not self.coordinator.targets_repository.chessboard_exists():
                self._pattern_preview.clear()
                self._pattern_info.setText("No chessboard configured")
                return
            chessboard = self.coordinator.targets_repository.load_chessboard()
            pixmap = render_chessboard_pixmap(chessboard, 120)
            self._pattern_preview.setPixmap(pixmap)
            squares_wide = chessboard.columns + 1
            squares_tall = chessboard.rows + 1
            self._pattern_info.setText(f"Chessboard: {squares_wide} x {squares_tall} squares")

        else:  # "charuco"
            if not self.coordinator.targets_repository.intrinsic_charuco_exists():
                self._pattern_preview.clear()
                self._pattern_info.setText("No charuco configured")
                return
            charuco = self.coordinator.targets_repository.load_intrinsic_charuco()
            pixmap = render_charuco_pixmap(charuco, 120)
            self._pattern_preview.setPixmap(pixmap)
            self._pattern_info.setText(f"ChArUco: {charuco.columns} x {charuco.rows}")

    def cleanup(self) -> None:
        """Clean up all presenters and widgets.

        Note: closeEvent is NOT reliable for tab widgets because
        removeTab() + deleteLater() doesn't trigger closeEvent.
        The parent (MainWidget) must call this during reload_workspace.
        """
        for cam_id, presenter in self._presenters.items():
            logger.info(f"Cleaning up presenter for cam {cam_id}")
            presenter.cleanup()

        for cam_id, widget in self._widgets.items():
            logger.info(f"Cleaning up widget for cam {cam_id}")
            self._content_layout.removeWidget(widget)
            widget.close()
            widget.deleteLater()

        self._presenters.clear()
        self._widgets.clear()
        self._current_cam_id = None

    def closeEvent(self, event) -> None:
        """Defensive cleanup on normal close."""
        self.cleanup()
        super().closeEvent(event)
