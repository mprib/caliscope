"""View for extrinsic calibration workflow.

Composes 3D visualization, calibration controls, filter UI,
coordinate frame controls, and quality panel. Follows the MVP pattern
with state-driven UI updates.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from caliscope.core.scale_accuracy import ScaleAccuracyData
from caliscope.gui.presenters.extrinsic_calibration_presenter import (
    ExtrinsicCalibrationPresenter,
    ExtrinsicCalibrationState,
    QualityPanelData,
)
from caliscope.gui.widgets.coverage_heatmap import CoverageHeatmapWidget
from caliscope.gui.widgets.quality_panel import QualityPanel
from caliscope.ui.viz.playback_triangulation_widget_pyvista import (
    PlaybackTriangulationWidgetPyVista,
)
from caliscope.ui.viz.playback_view_model import PlaybackViewModel

logger = logging.getLogger(__name__)


class ExtrinsicCalibrationView(QWidget):
    """View for extrinsic calibration workflow.

    Composes 3D visualization, calibration controls, filter UI,
    coordinate frame controls, and quality panel.

    UI state is derived from presenter state via _update_ui_for_state().
    This prevents state/UI divergence - there's no stored UI state.
    """

    def __init__(
        self,
        presenter: ExtrinsicCalibrationPresenter,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._presenter = presenter

        # Lazy-init: created on first view model update
        self._pyvista_widget: PlaybackTriangulationWidgetPyVista | None = None

        self._setup_ui()
        self._connect_signals()
        self._update_ui_for_state(presenter.state)

        # Show initial coverage before calibration
        self._presenter.emit_initial_coverage()

        logger.info("ExtrinsicCalibrationView created")

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the UI layout.

        Layout (top to bottom):
        1. Action bar (single button + progress) - workflow entry point
        2. 3D visualization
        3. Frame slider (directly under viz for consistency with reconstruction tab)
        4. Transform controls (filter, coordinate frame)
        5. Quality panel
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Row 1: Action bar at top (workflow entry point)
        action_bar = self._create_action_bar()
        main_layout.addWidget(action_bar)

        # Vertical splitter: 3D view + slider on top, controls below
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(self._splitter)

        # Top: 3D visualization + frame slider
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(4)
        self._viz_layout = viz_layout  # Store for lazy PyVista widget insertion

        # Placeholder shown until PyVista widget is created
        self._viz_placeholder = QLabel("Run calibration to see 3D visualization")
        self._viz_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._viz_placeholder.setStyleSheet(
            "QLabel { color: #888; font-size: 14px; "
            "background-color: #2a2a2a; border: 1px dashed #555; border-radius: 4px; }"
        )
        self._viz_placeholder.setFixedHeight(150)  # Compact, doesn't stretch
        viz_layout.addWidget(self._viz_placeholder)

        # Frame slider directly under 3D viz
        frame_widget = self._create_frame_slider()
        viz_layout.addWidget(frame_widget)
        self._splitter.addWidget(viz_container)

        # Bottom: transform controls + quality panel
        controls_panel = self._create_controls_panel()
        self._splitter.addWidget(controls_panel)

        # Initial sizes favor controls (placeholder is small)
        # Will be updated when PyVista widget is created
        self._splitter.setSizes([200, 600])
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 2)

    def _create_action_bar(self) -> QWidget:
        """Create the action bar with single action button + progress.

        The action button changes based on state:
        - NEEDS_BOOTSTRAP: "Calibrate"
        - OPTIMIZING: "Cancel"
        - CALIBRATED: "Re-optimize"
        """
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Single action button (text changes with state)
        self._action_btn = QPushButton("Calibrate")
        self._action_btn.setMinimumWidth(120)
        layout.addWidget(self._action_btn)

        # Progress section: bar first, then text
        progress_container = QWidget()
        progress_container.setMinimumWidth(450)  # Min width, can expand
        progress_layout = QHBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setFixedWidth(180)
        progress_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setMinimumWidth(250)  # Room for longer messages
        progress_layout.addWidget(self._progress_label, stretch=1)

        layout.addWidget(progress_container)
        self._progress_container = progress_container
        self._progress_container.hide()  # Hidden until optimizing

        layout.addStretch()
        return bar

    def _create_frame_slider(self) -> QWidget:
        """Create the frame slider widget."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(QLabel("Frame:"))

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.setEnabled(False)
        layout.addWidget(self._frame_slider, stretch=1)

        self._frame_display = QLabel("0 / 0")
        self._frame_display.setMinimumWidth(80)
        layout.addWidget(self._frame_display)

        return widget

    def _create_controls_panel(self) -> QWidget:
        """Create the bottom controls panel (coord frame, quality, filter)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Row 1: Coordinate frame controls
        self._coord_frame_group = QGroupBox("Coordinate Frame")
        coord_layout = QHBoxLayout(self._coord_frame_group)

        # Rotation buttons
        rotation_data = [
            ("X+", "x", 90),
            ("X−", "x", -90),
            ("Y+", "y", 90),
            ("Y−", "y", -90),
            ("Z+", "z", 90),
            ("Z−", "z", -90),
        ]

        for label, axis, degrees in rotation_data:
            btn = QPushButton(label)
            btn.setMaximumWidth(40)
            btn.clicked.connect(lambda checked, a=axis, d=degrees: self._on_rotate_clicked(a, d))
            coord_layout.addWidget(btn)

        coord_layout.addSpacing(20)

        self._set_origin_btn = QPushButton("Set Origin")
        self._set_origin_btn.setToolTip("Set world origin to charuco position at current frame")
        coord_layout.addWidget(self._set_origin_btn)

        coord_layout.addStretch()

        layout.addWidget(self._coord_frame_group)

        # Row 2: Quality panel (full width)
        self._quality_panel = QualityPanel()
        layout.addWidget(self._quality_panel)

        # Row 3: Filter controls (single line) + Coverage button
        self._filter_group = QGroupBox("Filter Outliers")
        filter_layout = QHBoxLayout(self._filter_group)

        filter_layout.addWidget(QLabel("Mode:"))
        self._filter_mode = QComboBox()
        self._filter_mode.addItems(["Percentile", "Absolute"])
        self._filter_mode.setMinimumWidth(100)
        filter_layout.addWidget(self._filter_mode)

        filter_layout.addWidget(QLabel("Value:"))
        self._filter_value = QDoubleSpinBox()
        self._filter_value.setRange(0.1, 100.0)
        self._filter_value.setValue(5.0)
        self._filter_value.setSuffix(" %")
        self._filter_value.setDecimals(1)
        filter_layout.addWidget(self._filter_value)

        self._filter_apply_btn = QPushButton("Apply")
        filter_layout.addWidget(self._filter_apply_btn)

        filter_layout.addSpacing(12)

        self._filter_preview_label = QLabel("Removes > ?.?? px")
        self._filter_preview_label.setStyleSheet("color: #888; font-style: italic;")
        filter_layout.addWidget(self._filter_preview_label)

        filter_layout.addStretch()

        # Coverage button opens floating dialog
        self._coverage_btn = QPushButton("View Coverage")
        self._coverage_btn.setToolTip("Show camera pair observation counts")
        self._coverage_btn.setEnabled(False)  # Enabled when coverage data arrives
        filter_layout.addWidget(self._coverage_btn)

        layout.addWidget(self._filter_group)

        # Coverage data stored for dialog (updated by signal)
        self._coverage_data: tuple[np.ndarray, list[str]] | None = None

        return panel

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Wire up signal connections between UI and presenter."""
        # Presenter → View
        self._presenter.state_changed.connect(self._update_ui_for_state)
        self._presenter.progress_updated.connect(self._on_progress_updated)
        self._presenter.quality_updated.connect(self._on_quality_updated)
        self._presenter.scale_accuracy_updated.connect(self._on_scale_accuracy_updated)
        self._presenter.coverage_updated.connect(self._on_coverage_updated)
        self._presenter.view_model_updated.connect(self._on_view_model_updated)

        # View → Presenter
        self._action_btn.clicked.connect(self._on_action_clicked)
        self._filter_apply_btn.clicked.connect(self._on_filter_apply_clicked)
        self._filter_mode.currentIndexChanged.connect(self._on_filter_mode_changed)
        self._filter_value.valueChanged.connect(self._on_filter_value_changed)
        self._set_origin_btn.clicked.connect(self._on_set_origin_clicked)
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self._coverage_btn.clicked.connect(self._show_coverage_dialog)

    # -------------------------------------------------------------------------
    # State-Driven UI
    # -------------------------------------------------------------------------

    def _update_ui_for_state(self, state: ExtrinsicCalibrationState) -> None:
        """Update all UI elements based on presenter state.

        Single handler that derives entire UI from current state.
        Prevents state/UI divergence - there's no stored UI state.

        Action button behavior:
        - NEEDS_BOOTSTRAP: "Calibrate" (enabled)
        - OPTIMIZING: "Cancel" (enabled)
        - CALIBRATED: "Re-optimize" (enabled)
        """
        is_running = state == ExtrinsicCalibrationState.OPTIMIZING
        has_bundle = state in (
            ExtrinsicCalibrationState.NEEDS_OPTIMIZATION,
            ExtrinsicCalibrationState.CALIBRATED,
        )

        # Single action button - text and behavior changes with state
        if state == ExtrinsicCalibrationState.NEEDS_BOOTSTRAP:
            self._action_btn.setText("Calibrate")
            self._action_btn.setEnabled(True)
        elif state == ExtrinsicCalibrationState.OPTIMIZING:
            self._action_btn.setText("Cancel")
            self._action_btn.setEnabled(True)
        elif state == ExtrinsicCalibrationState.CALIBRATED:
            self._action_btn.setText("Re-optimize")
            self._action_btn.setEnabled(True)
        else:  # NEEDS_OPTIMIZATION
            self._action_btn.setText("Optimize")
            self._action_btn.setEnabled(True)

        # Progress visibility (only when running)
        self._progress_container.setVisible(is_running)

        # Controls enabled only when we have bundle data
        self._filter_group.setEnabled(has_bundle)
        self._coord_frame_group.setEnabled(has_bundle)
        self._frame_slider.setEnabled(has_bundle)

        # Update filter preview when we have bundle
        if has_bundle:
            self._update_filter_preview()

        logger.debug(f"UI updated for state: {state}")

    # -------------------------------------------------------------------------
    # Slot Methods
    # -------------------------------------------------------------------------

    def _on_action_clicked(self) -> None:
        """Handle action button click - behavior depends on current state."""
        state = self._presenter.state

        if state == ExtrinsicCalibrationState.NEEDS_BOOTSTRAP:
            self._presenter.run_calibration()
        elif state == ExtrinsicCalibrationState.OPTIMIZING:
            self._presenter.cleanup()
        elif state == ExtrinsicCalibrationState.CALIBRATED:
            self._presenter.re_optimize()
        elif state == ExtrinsicCalibrationState.NEEDS_OPTIMIZATION:
            self._presenter.re_optimize()

    def _on_filter_apply_clicked(self) -> None:
        """Handle filter apply button click."""
        value = self._filter_value.value()
        if self._filter_mode.currentText() == "Percentile":
            self._presenter.filter_by_percentile(value)
        else:
            self._presenter.filter_by_threshold(value)

    def _on_filter_mode_changed(self, index: int) -> None:
        """Handle filter mode combo change."""
        is_percentile = index == 0
        self._filter_value.setSuffix(" %" if is_percentile else " px")
        if is_percentile:
            self._filter_value.setRange(0.1, 50.0)
            self._filter_value.setValue(5.0)
        else:
            self._filter_value.setRange(0.1, 10.0)
            self._filter_value.setValue(1.0)
        self._update_filter_preview()

    def _on_filter_value_changed(self, value: float) -> None:
        """Handle filter value spinbox change."""
        self._update_filter_preview()

    def _update_filter_preview(self) -> None:
        """Update the filter preview label based on current settings.

        Shows bidirectional translation:
        - Percentile mode: shows the pixel threshold
        - Absolute mode: shows the percentage that would be removed
        """
        preview_data = self._presenter.get_filter_preview()

        if preview_data.total_observations == 0:
            self._filter_preview_label.setText("No observations")
            return

        is_percentile = self._filter_mode.currentIndex() == 0
        value = self._filter_value.value()

        if is_percentile:
            # Percentile mode: show pixel threshold for this percentile
            int_pct = int(value)
            # Find closest available percentile
            if int_pct in preview_data.threshold_at_percentile:
                threshold = preview_data.threshold_at_percentile[int_pct]
                self._filter_preview_label.setText(f"Removes observations > {threshold:.2f} px")
            else:
                # Interpolate or show closest
                available = sorted(preview_data.threshold_at_percentile.keys())
                closest = min(available, key=lambda x: abs(x - int_pct))
                threshold = preview_data.threshold_at_percentile[closest]
                self._filter_preview_label.setText(f"~{closest}% removes > {threshold:.2f} px")
        else:
            # Absolute mode: show what percentage would be removed
            pct_removed = preview_data.percent_above_threshold(value)
            self._filter_preview_label.setText(f"Removes {pct_removed:.1f}% of observations")

    def _on_rotate_clicked(self, axis: str, degrees: float) -> None:
        """Handle rotation button click."""
        self._presenter.rotate(axis, degrees)

    def _on_set_origin_clicked(self) -> None:
        """Handle set origin button click."""
        sync_index = self._frame_slider.value()
        self._presenter.align_to_origin(sync_index)

    def _on_frame_slider_changed(self, value: int) -> None:
        """Handle frame slider value change.

        Directly updates the PyVista widget's sync_index for efficient rendering.
        Also updates presenter's internal tracking (for align_to_origin, etc.)
        but the presenter no longer emits view_model_updated for frame changes.
        """
        if self._pyvista_widget is not None:
            self._pyvista_widget.set_sync_index(value)
        # Keep presenter's internal state in sync (no signal emission)
        self._presenter.set_sync_index(value)
        self._update_frame_display()

    def _on_progress_updated(self, percent: int, message: str) -> None:
        """Handle progress update from presenter."""
        self._progress_bar.setValue(percent)
        self._progress_label.setText(message)

    def _on_quality_updated(self, data: QualityPanelData) -> None:
        """Handle quality data update from presenter."""
        self._quality_panel.set_reprojection_data(data)
        self._update_filter_preview()

    def _on_scale_accuracy_updated(self, data: ScaleAccuracyData) -> None:
        """Handle scale accuracy data update from presenter."""
        self._quality_panel.set_scale_accuracy(data)

    def _on_coverage_updated(self, coverage: np.ndarray, labels: list[str]) -> None:
        """Handle coverage data update from presenter."""
        self._coverage_data = (coverage, labels)
        # Enable button now that we have data
        self._coverage_btn.setEnabled(True)

    def _show_coverage_dialog(self) -> None:
        """Show floating dialog with coverage heatmap."""
        if self._coverage_data is None:
            return

        coverage, labels = self._coverage_data

        dialog = QDialog(self)
        dialog.setWindowTitle("Camera Coverage")
        dialog.setModal(False)  # Non-blocking

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)

        heatmap = CoverageHeatmapWidget()
        heatmap.set_data(coverage, killed_linkages=set(), labels=labels)
        layout.addWidget(heatmap)

        # Size based on camera count (widget sets its own minimum)
        dialog.adjustSize()
        dialog.show()

    def _on_view_model_updated(self, view_model: PlaybackViewModel) -> None:
        """Handle view model update from presenter."""
        # Create PyVista widget on first use (lazy initialization)
        if self._pyvista_widget is None:
            # Hide placeholder, show real widget
            self._viz_placeholder.hide()

            self._pyvista_widget = PlaybackTriangulationWidgetPyVista(view_model)
            self._pyvista_widget.show_playback_controls(False)  # We have our own slider
            # Insert at index 0 so it appears ABOVE the frame slider
            self._viz_layout.insertWidget(0, self._pyvista_widget)
            # PyVista widget should stretch to fill space
            self._viz_layout.setStretchFactor(self._pyvista_widget, 1)

            # Now that we have a 3D widget, give it more splitter space
            self._splitter.setSizes([700, 300])
            self._splitter.setStretchFactor(0, 3)
            self._splitter.setStretchFactor(1, 1)
        else:
            # Updating existing widget (coordinate transforms, re-optimization)
            # Preserve camera position so user doesn't lose their view
            self._pyvista_widget.set_view_model(view_model, preserve_camera=True)

        # Update slider range from view model
        n_frames = view_model.max_index - view_model.min_index + 1 if view_model.has_points else 0
        self._frame_slider.blockSignals(True)
        self._frame_slider.setMinimum(view_model.min_index if view_model.has_points else 0)
        self._frame_slider.setMaximum(view_model.max_index if view_model.has_points else 0)
        self._frame_slider.setValue(self._presenter.current_sync_index)
        self._frame_slider.setEnabled(n_frames > 1)
        self._frame_slider.blockSignals(False)

        self._update_frame_display()

    def _update_frame_display(self) -> None:
        """Update the frame counter display."""
        current = self._frame_slider.value()
        max_val = self._frame_slider.maximum()
        self._frame_display.setText(f"{current} / {max_val}")

    # -------------------------------------------------------------------------
    # VTK Lifecycle
    # -------------------------------------------------------------------------

    def suspend_vtk(self) -> None:
        """Pause VTK rendering when tab not active."""
        if self._pyvista_widget is not None:
            self._pyvista_widget.suspend_vtk()

    def resume_vtk(self) -> None:
        """Resume VTK rendering when tab becomes active."""
        if self._pyvista_widget is not None:
            self._pyvista_widget.resume_vtk()

    def cleanup(self) -> None:
        """Explicit cleanup - call before destruction."""
        if self._pyvista_widget is not None:
            self._pyvista_widget.close()
            self._pyvista_widget = None
        logger.info("ExtrinsicCalibrationView cleaned up")

    def closeEvent(self, event) -> None:
        """Handle close event."""
        self.cleanup()
        super().closeEvent(event)
