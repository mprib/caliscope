"""Widget for multi-camera synchronized video processing.

Displays camera thumbnails in a grid, progress bar, and control buttons.
Connects to MultiCameraProcessingPresenter for business logic.
"""

import logging
from typing import TYPE_CHECKING

from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from caliscope.gui.views.camera_thumbnail_card import CameraThumbnailCard
from caliscope.gui.widgets.coverage_heatmap import CoverageHeatmapWidget

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
    - Coverage summary after completion

    Note: This widget connects to a Presenter, not directly to the Coordinator.
    The Tab layer handles Presenter-to-Coordinator communication.
    """

    # Thumbnail layout constants
    MIN_CARD_WIDTH = 300  # Minimum width per card (thumbnail + margins)
    DEFAULT_COLUMNS = 2  # Fallback before first resize

    # Style for row labels with tooltip indicator (dotted underline)
    # QToolTip rule ensures tooltip popup doesn't inherit the underline
    _INFO_LABEL_STYLE = (
        "QLabel { text-decoration: underline dotted; text-decoration-color: #888; "
        "text-underline-offset: 2px; } "
        "QToolTip { text-decoration: none; }"
    )

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

    def _create_info_label(self, text: str, tooltip: str) -> QLabel:
        """Create a row label with discoverable tooltip styling (dotted underline)."""
        label = QLabel(text)
        label.setToolTip(tooltip)
        label.setStyleSheet(self._INFO_LABEL_STYLE)
        return label

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

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("Ready")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self._progress_label)

        layout.addWidget(progress_group)

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
        layout.addLayout(subsample_row)

        # Single action button (changes role based on state)
        self._action_btn = QPushButton("Start Processing")
        self._action_btn.clicked.connect(self._on_action_clicked)
        layout.addWidget(self._action_btn)

        # Coverage summary (hidden until complete)
        self._coverage_group = QGroupBox("Coverage Summary")
        coverage_outer_layout = QHBoxLayout(self._coverage_group)

        # Left side: heatmap visualization
        self._coverage_heatmap = CoverageHeatmapWidget()
        self._coverage_heatmap.setMinimumSize(200, 200)
        self._coverage_heatmap.setMaximumSize(300, 300)
        coverage_outer_layout.addWidget(self._coverage_heatmap)

        # Right side: text summary with tooltips
        coverage_text_layout = QFormLayout()
        coverage_text_layout.setHorizontalSpacing(12)

        # Frames processed (simple label, no tooltip needed)
        self._frames_value = QLabel("—")
        coverage_text_layout.addRow("Frames processed:", self._frames_value)

        # Network topology with tooltip
        topology_row = self._create_info_label(
            "Network topology:",
            "How cameras are connected via shared observations.\n\n"
            "Ring: Each camera shares points with neighbors\n"
            "Star: One central camera sees all others\n"
            "Mesh: All cameras share points with each other (ideal)",
        )
        self._topology_value = QLabel("—")
        coverage_text_layout.addRow(topology_row, self._topology_value)

        # Redundancy with tooltip
        redundancy_row = self._create_info_label(
            "Redundancy:",
            "How many independent paths connect cameras.\n\n"
            "1.0x = minimal (single path, fragile)\n"
            "1.5x+ = good redundancy\n"
            "2.0x+ = excellent (can lose links and still calibrate)",
        )
        self._redundancy_value = QLabel("—")
        coverage_text_layout.addRow(redundancy_row, self._redundancy_value)

        # Quality with tooltip
        quality_row = self._create_info_label(
            "Quality:",
            "Overall assessment based on topology, redundancy, and weak links.\n\n"
            "Good: Sufficient coverage for calibration\n"
            "Acceptable: May work but has weak areas\n"
            "Critical: Likely to fail - address issues before calibrating",
        )
        self._quality_value = QLabel("—")
        coverage_text_layout.addRow(quality_row, self._quality_value)

        # Weak links with tooltip
        weak_links_row = self._create_info_label(
            "Weak links:",
            "Camera pairs with insufficient shared observations.\n\n"
            "weak = 10-30 shared frames\n"
            "critical = under 10 frames\n"
            "none = no shared observations (cameras can't be connected)",
        )
        self._weak_links_value = QLabel("—")
        self._weak_links_value.setWordWrap(True)
        coverage_text_layout.addRow(weak_links_row, self._weak_links_value)

        coverage_outer_layout.addLayout(coverage_text_layout)

        # Guidance label (separate, full-width at bottom)
        self._guidance_label = QLabel("")
        self._guidance_label.setWordWrap(True)
        self._guidance_label.setStyleSheet("color: #FFA500;")  # Orange for warnings
        coverage_outer_layout.addWidget(self._guidance_label)

        self._coverage_group.hide()
        layout.addWidget(self._coverage_group)

        layout.addStretch()

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
            self._progress_label.setText("Waiting for configuration...")
            self._set_rotation_enabled(False)
            self._subsample_spin.setEnabled(False)
            self._coverage_group.hide()

        elif state == MultiCameraProcessingState.READY:
            self._action_btn.setText("Start Processing")
            self._action_btn.setEnabled(True)
            self._progress_label.setText("Ready to process")
            self._progress_bar.setValue(0)
            self._set_rotation_enabled(True)
            self._subsample_spin.setEnabled(True)
            self._coverage_group.hide()

        elif state == MultiCameraProcessingState.PROCESSING:
            self._action_btn.setText("Cancel")
            self._action_btn.setEnabled(True)
            self._set_rotation_enabled(False)
            self._subsample_spin.setEnabled(False)

        elif state == MultiCameraProcessingState.COMPLETE:
            self._action_btn.setText("Reset")
            self._action_btn.setEnabled(True)
            self._progress_label.setText("Processing complete")
            self._set_rotation_enabled(True)
            self._subsample_spin.setEnabled(True)
            self._coverage_group.show()

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
        from caliscope.core.coverage_analysis import (
            LinkQuality,
            generate_multi_camera_guidance,
        )

        # Update coverage heatmap with port-based labels
        ports = sorted(self._presenter.cameras.keys())
        labels = [f"C{p}" for p in ports]
        self._coverage_heatmap.set_data(
            coverage_report.pairwise_frames,
            killed_linkages=set(),  # No killed linkages in initial processing
            labels=labels,
        )

        # Frame count from ImagePoints
        n_frames = image_points.df["sync_index"].nunique()
        self._frames_value.setText(str(n_frames))

        # Network topology
        topology = coverage_report.topology_class.value.title()
        self._topology_value.setText(topology)

        # Redundancy factor (1.0 = minimal tree, higher = more robust)
        redundancy = coverage_report.redundancy_factor
        if redundancy >= 2.0:
            redundancy_text = f"{redundancy:.1f}x (excellent)"
        elif redundancy >= 1.5:
            redundancy_text = f"{redundancy:.1f}x (good)"
        else:
            redundancy_text = f"{redundancy:.1f}x (minimal)"
        self._redundancy_value.setText(redundancy_text)

        # Quality assessment
        if coverage_report.has_critical_issues:
            if coverage_report.isolated_cameras:
                quality = "Critical - isolated cameras detected"
            else:
                quality = "Critical - weak camera links"
        elif coverage_report.weak_links:
            quality = f"Acceptable ({len(coverage_report.weak_links)} weak links)"
        else:
            quality = "Good"
        self._quality_value.setText(quality)

        # Weak links detail
        if coverage_report.weak_links:
            weak_text_parts = []
            for cam_a, cam_b, link_quality in coverage_report.weak_links[:5]:
                quality_str = link_quality.value if link_quality != LinkQuality.DISCONNECTED else "none"
                weak_text_parts.append(f"C{cam_a}↔C{cam_b}: {quality_str}")
            weak_text = ", ".join(weak_text_parts)
            if len(coverage_report.weak_links) > 5:
                weak_text += f" (+{len(coverage_report.weak_links) - 5} more)"
            self._weak_links_value.setText(weak_text)
        else:
            self._weak_links_value.setText("None")

        # Guidance messages
        guidance = generate_multi_camera_guidance(coverage_report)
        if guidance:
            self._guidance_label.setText("\n".join(guidance))
        else:
            self._guidance_label.setText("")

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
