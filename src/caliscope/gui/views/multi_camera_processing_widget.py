"""Widget for multi-camera synchronized video processing.

Displays camera thumbnails in a grid, progress bar, and control buttons.
Connects to MultiCameraProcessingPresenter for business logic.
"""

import logging
from typing import TYPE_CHECKING

from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from caliscope.gui.theme import Styles
from caliscope.gui.views.camera_thumbnail_card import CameraThumbnailCard
from caliscope.gui.widgets.coverage_heatmap import CoverageHeatmapWidget
from caliscope.gui.widgets.structural_warnings import StructuralWarningsWidget

if TYPE_CHECKING:
    from caliscope.core.coverage_analysis import ExtrinsicCoverageReport
    from caliscope.core.point_data import ImagePoints
    from caliscope.gui.presenters.multi_camera_processing_presenter import (
        MultiCameraProcessingPresenter,
        MultiCameraProcessingState,
    )
    from caliscope.packets import PointPacket

logger = logging.getLogger(__name__)


class MultiCameraProcessingWidget(QWidget):
    """View for multi-camera processing workflow.

    Displays:
    - Grid of camera thumbnails with rotation controls
    - Progress bar during processing
    - Start/Cancel/Reset buttons
    - Coverage heatmap and structural warnings after completion

    Note: This widget connects to a Presenter, not directly to the Coordinator.
    The Tab layer handles Presenter-to-Coordinator communication.
    """

    # Thumbnail layout constants
    MIN_CARD_WIDTH = 300  # Minimum width per card (thumbnail + margins)
    DEFAULT_COLUMNS = 2  # Fallback before first resize

    def __init__(
        self,
        presenter: "MultiCameraProcessingPresenter",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._camera_cards: dict[int, CameraThumbnailCard] = {}
        self._current_columns = self.DEFAULT_COLUMNS

        self._setup_ui()
        self._connect_signals()
        self._update_ui_for_state(presenter.state)

    def _setup_ui(self) -> None:
        """Build the UI layout."""
        layout = QVBoxLayout(self)

        # Camera grid in a scroll area (so it doesn't push controls off screen)
        self._camera_grid = QGridLayout()
        self._camera_group = QGroupBox("Cameras")
        self._camera_group.setLayout(self._camera_grid)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setWidget(self._camera_group)
        self._scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        layout.addWidget(self._scroll_area, stretch=1)  # Takes available space

        # Create cards for existing cameras
        self._rebuild_camera_grid()

        # Bottom section: Controls (left) | Coverage Summary (right)
        bottom_section = QHBoxLayout()
        bottom_section.setSpacing(24)

        # === Left side: Controls ===
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Subsample control row
        subsample_row = QHBoxLayout()
        subsample_row.addWidget(QLabel("Process every"))
        self._subsample_spin = QSpinBox()
        self._subsample_spin.setRange(1, 20)
        self._subsample_spin.setValue(5)
        self._subsample_spin.setToolTip(
            "Skip frames to speed up processing.\n\n"
            "1 = every frame (slow, most data)\n"
            "5 = every 5th frame (default, good balance)\n"
            "20 = every 20th frame (fast, less data)"
        )
        subsample_row.addWidget(self._subsample_spin)
        subsample_row.addWidget(QLabel("frames"))
        subsample_row.addStretch()
        controls_layout.addLayout(subsample_row)

        controls_layout.addSpacing(12)

        # Action button (centered, content-fit width)
        button_row = QHBoxLayout()
        button_row.addStretch()
        self._action_btn = QPushButton("Start Processing")
        self._action_btn.setStyleSheet(Styles.PRIMARY_BUTTON)
        self._action_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._action_btn.clicked.connect(self._on_action_clicked)
        button_row.addWidget(self._action_btn)
        button_row.addStretch()
        controls_layout.addLayout(button_row)

        controls_layout.addSpacing(12)

        # Progress section (shown only during/after processing)
        self._progress_container = QWidget()
        progress_layout = QVBoxLayout(self._progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("Ready")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setStyleSheet("color: #888; font-size: 12px;")
        progress_layout.addWidget(self._progress_label)

        controls_layout.addWidget(self._progress_container)
        controls_layout.addStretch()

        bottom_section.addWidget(controls_container, stretch=1)

        # === Right side: Coverage Summary (always visible) ===
        self._coverage_container = QFrame()
        self._coverage_container.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border-radius: 4px;
            }
        """)
        coverage_layout = QVBoxLayout(self._coverage_container)
        coverage_layout.setContentsMargins(16, 16, 16, 16)

        # Coverage title
        coverage_title = QLabel("Coverage Summary")
        coverage_title.setStyleSheet("font-weight: bold; font-size: 13px; background: transparent;")
        coverage_layout.addWidget(coverage_title)

        coverage_layout.addSpacing(8)

        # Stacked content: placeholder OR actual coverage data
        # We use a simple show/hide approach with two containers

        # Placeholder (shown before processing)
        self._coverage_placeholder = QWidget()
        placeholder_layout = QVBoxLayout(self._coverage_placeholder)
        placeholder_layout.setContentsMargins(0, 0, 0, 0)
        placeholder_layout.addStretch()
        placeholder_text = QLabel("Process video to see\ncamera coverage analysis")
        placeholder_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_text.setStyleSheet("color: #666; font-size: 12px; background: transparent;")
        placeholder_layout.addWidget(placeholder_text)
        placeholder_layout.addStretch()
        coverage_layout.addWidget(self._coverage_placeholder)

        # Actual coverage content (shown after processing)
        self._coverage_content = QWidget()
        content_layout = QHBoxLayout(self._coverage_content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Left: heatmap visualization
        heatmap_layout = QVBoxLayout()
        heatmap_label = QLabel("Shared Point Observations")
        heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heatmap_label.setStyleSheet("color: #888; font-size: 11px; background: transparent;")
        heatmap_layout.addWidget(heatmap_label)

        self._coverage_heatmap = CoverageHeatmapWidget()
        self._coverage_heatmap.setMinimumSize(180, 180)
        self._coverage_heatmap.setMaximumSize(250, 250)
        heatmap_layout.addWidget(self._coverage_heatmap)
        content_layout.addLayout(heatmap_layout)

        # Right: frames processed + structural warnings
        right_side_layout = QVBoxLayout()

        frames_row = QHBoxLayout()
        frames_label = QLabel("Frames processed:")
        frames_label.setStyleSheet("background: transparent;")
        frames_row.addWidget(frames_label)
        self._frames_value = QLabel("—")
        self._frames_value.setStyleSheet("background: transparent;")
        frames_row.addWidget(self._frames_value)
        frames_row.addStretch()
        right_side_layout.addLayout(frames_row)

        self._warnings_widget = StructuralWarningsWidget()
        right_side_layout.addWidget(self._warnings_widget)
        right_side_layout.addStretch()

        content_layout.addLayout(right_side_layout)

        self._coverage_content.hide()  # Start hidden
        coverage_layout.addWidget(self._coverage_content)

        # Set minimum size so layout doesn't jump
        self._coverage_container.setMinimumSize(350, 200)

        bottom_section.addWidget(self._coverage_container, stretch=1)

        layout.addLayout(bottom_section)

    def _rebuild_camera_grid(self) -> None:
        """Rebuild the camera grid from presenter's cameras."""
        # Clear existing cards
        for card in self._camera_cards.values():
            self._camera_grid.removeWidget(card)
            card.deleteLater()
        self._camera_cards.clear()

        # Create new cards
        cameras = self._presenter.cameras
        for i, port in enumerate(sorted(cameras.keys())):
            row = i // self._current_columns
            col = i % self._current_columns

            card = CameraThumbnailCard(port)
            card.rotate_requested.connect(self._on_rotate_requested)
            self._camera_grid.addWidget(card, row, col)
            self._camera_cards[port] = card

        # Load initial thumbnails
        for port, frame in self._presenter.thumbnails.items():
            if port in self._camera_cards:
                rotation = self._presenter.cameras[port].rotation_count
                self._camera_cards[port].set_thumbnail(frame, rotation)

    def _reflow_grid(self) -> None:
        """Reposition existing camera cards based on current column count.

        More efficient than _rebuild_camera_grid - doesn't recreate widgets.
        """
        ports = sorted(self._camera_cards.keys())
        for i, port in enumerate(ports):
            card = self._camera_cards[port]
            row = i // self._current_columns
            col = i % self._current_columns
            self._camera_grid.addWidget(card, row, col)

    def resizeEvent(self, event) -> None:
        """Recalculate grid columns when widget width changes."""
        super().resizeEvent(event)

        # Calculate how many columns fit
        available_width = self._scroll_area.viewport().width()
        new_columns = max(1, available_width // self.MIN_CARD_WIDTH)

        # Only reflow if column count changed
        if new_columns != self._current_columns and self._camera_cards:
            self._current_columns = new_columns
            self._reflow_grid()

    def _connect_signals(self) -> None:
        """Connect presenter signals to view slots."""
        self._presenter.state_changed.connect(self._update_ui_for_state)
        self._presenter.progress_updated.connect(self._on_progress_updated)
        self._presenter.thumbnail_updated.connect(self._on_thumbnail_updated)
        self._presenter.processing_complete.connect(self._on_processing_complete)
        self._presenter.processing_failed.connect(self._on_processing_failed)

    def _update_ui_for_state(self, state: "MultiCameraProcessingState") -> None:
        """Update UI elements based on presenter state."""
        from caliscope.gui.presenters.multi_camera_processing_presenter import (
            MultiCameraProcessingState,
        )

        logger.debug(f"Updating UI for state: {state}")

        if state == MultiCameraProcessingState.UNCONFIGURED:
            self._action_btn.setText("Start Processing")
            self._action_btn.setEnabled(False)
            self._progress_container.hide()
            self._set_rotation_enabled(False)
            self._subsample_spin.setEnabled(False)
            # Show placeholder, hide content
            self._coverage_placeholder.show()
            self._coverage_content.hide()

        elif state == MultiCameraProcessingState.READY:
            self._action_btn.setText("Start Processing")
            self._action_btn.setEnabled(True)
            self._progress_container.hide()
            self._set_rotation_enabled(True)
            self._subsample_spin.setEnabled(True)
            # Show placeholder, hide content
            self._coverage_placeholder.show()
            self._coverage_content.hide()

        elif state == MultiCameraProcessingState.PROCESSING:
            self._action_btn.setText("Cancel")
            self._action_btn.setEnabled(True)
            self._progress_container.show()
            self._progress_bar.setValue(0)
            self._progress_label.setText("Starting...")
            self._set_rotation_enabled(False)
            self._subsample_spin.setEnabled(False)
            # Keep showing placeholder during processing
            self._coverage_placeholder.show()
            self._coverage_content.hide()

        elif state == MultiCameraProcessingState.COMPLETE:
            self._action_btn.setText("Reset")
            self._action_btn.setEnabled(True)
            self._progress_container.show()
            self._progress_label.setText("Processing complete")
            self._set_rotation_enabled(True)
            self._subsample_spin.setEnabled(True)
            # Show content, hide placeholder
            self._coverage_placeholder.hide()
            self._coverage_content.show()

    def _set_rotation_enabled(self, enabled: bool) -> None:
        """Enable or disable rotation controls on all camera cards."""
        for card in self._camera_cards.values():
            card.set_enabled(enabled)

    # -------------------------------------------------------------------------
    # Slots for Presenter Signals
    # -------------------------------------------------------------------------

    def _on_progress_updated(self, current: int, total: int, percent: int) -> None:
        """Handle progress update from presenter."""
        self._progress_bar.setValue(percent)
        self._progress_label.setText(f"Processing: {current}/{total} frames ({percent}%)")

    def _on_thumbnail_updated(self, port: int, frame: NDArray, points: "PointPacket | None") -> None:
        """Handle thumbnail update from presenter.

        Args:
            port: Camera port
            frame: BGR image
            points: Tracked landmarks to overlay (or None)
        """
        if port in self._camera_cards:
            rotation = self._presenter.cameras[port].rotation_count
            self._camera_cards[port].set_thumbnail(frame, rotation, points)

    def _on_processing_complete(
        self,
        image_points: "ImagePoints",
        coverage_report: "ExtrinsicCoverageReport",
        tracker: object,  # Tracker included in signal; widget doesn't use it (Tab does)
    ) -> None:
        """Handle processing completion from presenter."""
        from caliscope.core.coverage_analysis import detect_structural_warnings

        # Update coverage heatmap with port-based labels
        ports = sorted(self._presenter.cameras.keys())
        labels = [f"C{p}" for p in ports]
        self._coverage_heatmap.set_data(
            coverage_report.pairwise_observations,
            killed_linkages=set(),  # No killed linkages in initial processing
            labels=labels,
        )

        # Frame count from ImagePoints
        n_frames = image_points.df["sync_index"].nunique()
        self._frames_value.setText(str(n_frames))

        # Detect and display structural warnings
        n_cameras = len(ports)
        warnings = detect_structural_warnings(coverage_report, n_cameras)
        self._warnings_widget.set_warnings(warnings)

    def _on_processing_failed(self, error_msg: str) -> None:
        """Handle processing failure from presenter."""
        self._progress_label.setText(f"Failed: {error_msg}")

    # -------------------------------------------------------------------------
    # Slots for UI Actions
    # -------------------------------------------------------------------------

    def _on_action_clicked(self) -> None:
        """Handle action button click - routes based on current state."""
        from caliscope.gui.presenters.multi_camera_processing_presenter import (
            MultiCameraProcessingState,
        )

        state = self._presenter.state
        if state == MultiCameraProcessingState.READY:
            subsample = self._subsample_spin.value()
            logger.info(f"Starting processing with subsample={subsample}")
            self._presenter.start_processing(subsample=subsample)
        elif state == MultiCameraProcessingState.PROCESSING:
            self._presenter.cancel_processing()
        elif state == MultiCameraProcessingState.COMPLETE:
            self._presenter.reset()

    def _on_rotate_requested(self, port: int, direction: int) -> None:
        """Handle rotation request from camera card.

        Args:
            port: Camera port
            direction: +1 for clockwise, -1 for counter-clockwise
        """
        if port in self._presenter.cameras:
            current = self._presenter.cameras[port].rotation_count
            new_rotation = (current + direction) % 4
            self._presenter.set_rotation(port, new_rotation)
