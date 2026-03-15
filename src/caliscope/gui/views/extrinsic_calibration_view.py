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
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from caliscope.core.scale_accuracy import VolumetricScaleReport
from caliscope.gui.presenters.extrinsic_calibration_presenter import (
    ExtrinsicCalibrationPresenter,
    ExtrinsicCalibrationState,
    QualityPanelData,
)
from caliscope.gui.theme import Styles
from caliscope.gui.view_models.playback_view_model import PlaybackViewModel
from caliscope.gui.widgets.coverage_heatmap import CoverageHeatmapWidget
from caliscope.gui.widgets.playback_viz_widget import PlaybackVizWidget
from caliscope.gui.widgets.quality_panel import QualityPanel
from caliscope.gui.widgets.scale_detail_dialog import ScaleDetailDialog
from caliscope.gui.widgets.scale_sparkline import ScaleSparkline

logger = logging.getLogger(__name__)


class ExtrinsicCalibrationView(QWidget):
    """View for extrinsic calibration workflow.

    Composes 3D visualization, calibration controls, filter UI,
    coordinate frame controls, and quality panel.

    UI state is derived from presenter state via _update_ui_for_state().
    This prevents state/UI divergence - there's no stored UI state.

    Layout (top to bottom):
    1. Primary actions (split layout):
       - Left: Calibrate/Recalibrate + progress
       - Right: Set Origin + View Coverage
    2-6. Calibrated controls (progressive disclosure - hidden until CALIBRATED):
       2. Frame slider (full width)
       3. Rotation buttons (2x3 grid, left-aligned)
       4. Scale accuracy sparkline + expand button
       5. Quality panel (3 sections, evenly distributed)
       6. Filter controls (at bottom)
    """

    def __init__(
        self,
        presenter: ExtrinsicCalibrationPresenter,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._presenter = presenter

        # Lazy-init: created on first view model update WHEN VISIBLE
        # This prevents VTK OpenGL context issues when tab starts disabled
        self._pyvista_widget: PlaybackVizWidget | None = None
        self._pending_view_model: PlaybackViewModel | None = None

        # Valid sync indices for frame navigation (sparse data support)
        # Slider position = index into this array, not the actual sync_index
        self._valid_sync_indices: np.ndarray = np.array([], dtype=np.int64)

        # Scale accuracy state
        self._scale_detail_dialog: ScaleDetailDialog | None = None
        self._latest_scale_report: VolumetricScaleReport | None = None

        self._setup_ui()
        self._connect_signals()
        self._update_ui_for_state(presenter.state)

        # Emit initial state (coverage, and if restored session: quality + 3D viz)
        self._presenter.emit_initial_state()

        logger.info("ExtrinsicCalibrationView created")

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Vertical splitter: 3D view on top, controls below
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(self._splitter)

        # Top: 3D visualization
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(0)
        self._viz_layout = viz_layout

        # Placeholder shown until PyVista widget is created
        self._viz_placeholder = QLabel("Run calibration to see 3D visualization")
        self._viz_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._viz_placeholder.setStyleSheet(
            "QLabel { color: #888; font-size: 14px; "
            "background-color: #2a2a2a; border: 1px dashed #555; border-radius: 4px; }"
        )
        self._viz_placeholder.setFixedHeight(150)
        viz_layout.addWidget(self._viz_placeholder)
        self._splitter.addWidget(viz_container)

        # Bottom: Controls panel
        controls_panel = self._create_controls_panel()
        self._splitter.addWidget(controls_panel)

        # Initial sizes favor controls (placeholder is small)
        self._splitter.setSizes([200, 400])
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 2)

    def _create_controls_panel(self) -> QWidget:
        """Create the bottom controls panel.

        Layout order:
        1. Primary actions row (Calibrate/Recalibrate + progress + Set Origin + View Coverage)
        2-6. Calibrated controls (hidden until CALIBRATED):
           2. Frame slider
           3. Rotation buttons
           4. Sparkline
           5. Quality panel
           6. Filter controls
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.setSpacing(12)

        # Row 1: Primary actions - split layout
        primary_row = self._create_primary_actions_row()
        layout.addWidget(primary_row)

        # Rows 2-6: Calibrated controls (hidden until CALIBRATED)
        self._calibrated_controls = QWidget()
        cal_layout = QVBoxLayout(self._calibrated_controls)
        cal_layout.setContentsMargins(0, 0, 0, 0)
        cal_layout.setSpacing(12)

        # Row 2: Frame slider (full width)
        slider_row = self._create_slider_row()
        cal_layout.addWidget(slider_row)

        # Row 3: Rotation buttons (compact, left-aligned)
        rotation_row = self._create_rotation_row()
        cal_layout.addWidget(rotation_row)

        # Row 4: Sparkline + expand button
        sparkline_row = self._create_sparkline_row()
        cal_layout.addWidget(sparkline_row)

        # Row 5: Quality panel
        self._quality_panel = QualityPanel()
        cal_layout.addWidget(self._quality_panel)

        # Row 6: Filter controls
        filter_row = self._create_filter_row()
        cal_layout.addWidget(filter_row)

        layout.addWidget(self._calibrated_controls)
        self._calibrated_controls.hide()  # Hidden until CALIBRATED

        # Coverage data stored for dialog (updated by signal)
        self._coverage_data: tuple[np.ndarray, list[str]] | None = None

        return panel

    def _create_primary_actions_row(self) -> QWidget:
        """Create primary actions row with split layout.

        Left: Calibrate/Recalibrate + progress
        Right: Set Origin + View Coverage
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Left side: primary action
        self._action_btn = QPushButton("Calibrate")
        self._action_btn.setStyleSheet(Styles.PRIMARY_BUTTON)
        layout.addWidget(self._action_btn)

        # Progress section (shown during optimization)
        self._progress_container = QWidget()
        progress_layout = QHBoxLayout(self._progress_container)
        progress_layout.setContentsMargins(16, 0, 0, 0)
        progress_layout.setSpacing(8)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setFixedWidth(150)
        progress_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setMinimumWidth(200)
        progress_layout.addWidget(self._progress_label)

        layout.addWidget(self._progress_container)
        self._progress_container.hide()

        layout.addStretch()  # Push right-side buttons to the right

        # Right side: secondary actions
        self._set_origin_btn = QPushButton("Set Origin")
        self._set_origin_btn.setToolTip("Set world origin to charuco position at current frame")
        self._set_origin_btn.setStyleSheet(Styles.GHOST_BUTTON)
        self._set_origin_btn.setVisible(False)  # Hidden until calibrated
        layout.addWidget(self._set_origin_btn)

        self._coverage_btn = QPushButton("View Coverage")
        self._coverage_btn.setToolTip("Show camera pair observation counts")
        self._coverage_btn.setEnabled(False)
        self._coverage_btn.setStyleSheet(Styles.GHOST_BUTTON)
        layout.addWidget(self._coverage_btn)

        return row

    def _create_slider_row(self) -> QWidget:
        """Create frame slider row (full width)."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(QLabel("Frame:"))

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.setEnabled(False)
        self._frame_slider.setStyleSheet(Styles.SLIDER)
        layout.addWidget(self._frame_slider, stretch=1)

        self._frame_display = QLabel("0 / 0")
        self._frame_display.setMinimumWidth(70)
        layout.addWidget(self._frame_display)

        return row

    def _create_rotation_row(self) -> QWidget:
        """Create rotation buttons row (left-aligned, compact)."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(QLabel("Rotate:"))
        rotation_grid = self._create_rotation_grid()
        layout.addWidget(rotation_grid)
        layout.addStretch()

        return row

    def _create_rotation_grid(self) -> QWidget:
        """Create 2x3 grid of rotation buttons.

        Layout:
        X+  Y+  Z+
        X-  Y-  Z-
        """
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        # Rotation data: (label, axis, degrees, row, col, bg_color, hover_color)
        rotation_data = [
            ("X+", "x", 90, 0, 0, "#cc3333", "#ff4444"),  # Red
            ("Y+", "y", 90, 0, 1, "#33aa33", "#44cc44"),  # Green
            ("Z+", "z", 90, 0, 2, "#3366cc", "#4488ff"),  # Blue
            ("X-", "x", -90, 1, 0, "#cc3333", "#ff4444"),  # Red
            ("Y-", "y", -90, 1, 1, "#33aa33", "#44cc44"),  # Green
            ("Z-", "z", -90, 1, 2, "#3366cc", "#4488ff"),  # Blue
        ]

        self._rotation_btns: list[QPushButton] = []
        for label, axis, degrees, row, col, bg_color, hover_color in rotation_data:
            btn = QPushButton(label)
            btn.setFixedSize(36, 28)
            btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {bg_color};
                    color: white;
                    border: none;
                    border-radius: 3px;
                    font-weight: bold;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: {hover_color};
                }}
                QPushButton:disabled {{
                    background-color: #555;
                    color: #888;
                }}
                """
            )
            btn.clicked.connect(lambda checked, a=axis, d=degrees: self._on_rotate_clicked(a, d))
            grid.addWidget(btn, row, col)
            self._rotation_btns.append(btn)

        return widget

    def _create_sparkline_row(self) -> QWidget:
        """Create sparkline chart + expand button row."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Sparkline (stretches to fill available space)
        self._sparkline = ScaleSparkline()
        layout.addWidget(self._sparkline, stretch=1)

        # Expand button (20x20, icon-only, hidden until data available)
        self._sparkline_expand_btn = QToolButton()
        self._sparkline_expand_btn.setText("⊕")
        self._sparkline_expand_btn.setFixedSize(20, 20)
        self._sparkline_expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sparkline_expand_btn.setToolTip("Show detailed scale accuracy chart")
        self._sparkline_expand_btn.hide()
        layout.addWidget(self._sparkline_expand_btn)

        return row

    def _create_filter_row(self) -> QWidget:
        """Create filter controls row (at bottom)."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Filter label
        filter_label = QLabel("Filter Outliers:")
        filter_label.setStyleSheet("font-weight: bold; color: #aaa;")
        layout.addWidget(filter_label)

        layout.addWidget(QLabel("Mode:"))
        self._filter_mode = QComboBox()
        self._filter_mode.addItems(["Percentile", "Absolute"])
        self._filter_mode.setMinimumWidth(90)
        layout.addWidget(self._filter_mode)

        layout.addWidget(QLabel("Value:"))
        self._filter_value = QDoubleSpinBox()
        self._filter_value.setRange(0.1, 100.0)
        self._filter_value.setValue(5.0)
        self._filter_value.setSuffix(" %")
        self._filter_value.setDecimals(1)
        self._filter_value.setMinimumWidth(70)
        layout.addWidget(self._filter_value)

        # Apply button (ghost style)
        self._filter_apply_btn = QPushButton("Apply")
        self._filter_apply_btn.setStyleSheet(Styles.GHOST_BUTTON)
        layout.addWidget(self._filter_apply_btn)

        layout.addSpacing(8)

        self._filter_preview_label = QLabel("Removes > ?.?? px")
        self._filter_preview_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._filter_preview_label)

        layout.addStretch()

        return row

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Wire up signal connections between UI and presenter."""
        # Presenter -> View
        self._presenter.state_changed.connect(self._update_ui_for_state)
        self._presenter.progress_updated.connect(self._on_progress_updated)
        self._presenter.quality_updated.connect(self._on_quality_updated)
        self._presenter.volumetric_accuracy_updated.connect(self._on_volumetric_accuracy_updated)
        self._presenter.coverage_updated.connect(self._on_coverage_updated)
        self._presenter.view_model_updated.connect(self._on_view_model_updated)

        # View -> Presenter
        self._action_btn.clicked.connect(self._on_action_clicked)
        self._filter_apply_btn.clicked.connect(self._on_filter_apply_clicked)
        self._filter_mode.currentIndexChanged.connect(self._on_filter_mode_changed)
        self._filter_value.valueChanged.connect(self._on_filter_value_changed)
        self._set_origin_btn.clicked.connect(self._on_set_origin_clicked)
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self._coverage_btn.clicked.connect(self._show_coverage_dialog)

        # Sparkline interactions (cursor sync happens in _on_frame_slider_changed, not direct connection)
        self._sparkline.frame_clicked.connect(self._on_sparkline_frame_clicked)
        self._sparkline_expand_btn.clicked.connect(self._show_scale_detail_dialog)

    # -------------------------------------------------------------------------
    # State-Driven UI
    # -------------------------------------------------------------------------

    def _update_ui_for_state(self, state: ExtrinsicCalibrationState) -> None:
        """Update all UI elements based on presenter state."""
        is_running = state == ExtrinsicCalibrationState.CALIBRATING
        has_capture_volume = state == ExtrinsicCalibrationState.CALIBRATED

        # Action button text changes based on state
        if is_running:
            self._action_btn.setText("Cancel")
        elif has_capture_volume:
            self._action_btn.setText("Recalibrate")
        else:
            self._action_btn.setText("Calibrate")
        self._action_btn.setEnabled(True)

        # Set Origin only visible when calibrated
        self._set_origin_btn.setVisible(has_capture_volume)

        # Progress only when running
        self._progress_container.setVisible(is_running)

        # Progressive disclosure: show calibrated controls when we have a capture volume.
        # During filter re-optimization (CALIBRATING with capture volume preserved),
        # controls stay visible. Only fresh calibration (capture volume cleared) hides them.
        show_calibrated = has_capture_volume or (is_running and self._presenter.capture_volume is not None)
        self._calibrated_controls.setVisible(show_calibrated)

        # Controls enabled when calibrated and not running
        # During filter re-opt, controls are visible but disabled
        controls_enabled = has_capture_volume and not is_running
        self._frame_slider.setEnabled(controls_enabled)
        self._filter_apply_btn.setEnabled(controls_enabled)
        self._filter_mode.setEnabled(controls_enabled)
        self._filter_value.setEnabled(controls_enabled)
        for btn in self._rotation_btns:
            btn.setEnabled(controls_enabled)

        # Update filter preview when we have a capture volume
        if has_capture_volume:
            self._update_filter_preview()

        logger.debug(f"UI updated for state: {state}")

    # -------------------------------------------------------------------------
    # Slot Methods
    # -------------------------------------------------------------------------

    def _on_action_clicked(self) -> None:
        """Handle action button click - behavior depends on current state."""
        state = self._presenter.state

        if state == ExtrinsicCalibrationState.CALIBRATING:
            self._presenter.cancel_calibration()
        elif state == ExtrinsicCalibrationState.CALIBRATED:
            # Confirm before discarding current calibration
            if self._confirm_recalibrate():
                self._presenter.run_calibration()
        else:  # NEEDS_CALIBRATION
            self._presenter.run_calibration()

    def _confirm_recalibrate(self) -> bool:
        """Show confirmation dialog before discarding current calibration."""
        result = QMessageBox.warning(
            self,
            "Discard current calibration?",
            "This will discard the current calibration and run a fresh calibration from scratch.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return result == QMessageBox.StandardButton.Ok

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
        """Update the filter preview label based on current settings."""
        preview_data = self._presenter.get_filter_preview()

        if preview_data.total_observations == 0:
            self._filter_preview_label.setText("No observations")
            return

        is_percentile = self._filter_mode.currentIndex() == 0
        value = self._filter_value.value()

        if is_percentile:
            int_pct = int(value)
            if int_pct in preview_data.threshold_at_percentile:
                threshold = preview_data.threshold_at_percentile[int_pct]
                self._filter_preview_label.setText(f"Removes > {threshold:.2f} px")
            else:
                available = sorted(preview_data.threshold_at_percentile.keys())
                closest = min(available, key=lambda x: abs(x - int_pct))
                threshold = preview_data.threshold_at_percentile[closest]
                self._filter_preview_label.setText(f"~{closest}% > {threshold:.2f} px")
        else:
            pct_removed = preview_data.percent_above_threshold(value)
            self._filter_preview_label.setText(f"Removes {pct_removed:.1f}%")

    def _on_rotate_clicked(self, axis: str, degrees: float) -> None:
        """Handle rotation button click."""
        self._presenter.rotate(axis, degrees)

    def _on_set_origin_clicked(self) -> None:
        """Handle set origin button click."""
        if len(self._valid_sync_indices) == 0:
            return
        slider_position = self._frame_slider.value()
        actual_sync_index = int(self._valid_sync_indices[slider_position])
        self._presenter.align_to_origin(actual_sync_index)

    def _on_frame_slider_changed(self, value: int) -> None:
        """Handle frame slider value change."""
        if len(self._valid_sync_indices) == 0:
            return

        actual_sync_index = int(self._valid_sync_indices[value])

        if self._pyvista_widget is not None:
            self._pyvista_widget.set_sync_index(actual_sync_index)
        self._presenter.set_sync_index(actual_sync_index)
        self._sparkline.set_cursor(actual_sync_index)
        if self._scale_detail_dialog is not None:
            self._scale_detail_dialog.set_cursor(actual_sync_index)
        self._update_frame_display()

    def _on_progress_updated(self, percent: int, message: str) -> None:
        """Handle progress update from presenter."""
        self._progress_bar.setValue(percent)
        self._progress_label.setText(message)

    def _on_quality_updated(self, data: QualityPanelData) -> None:
        """Handle quality data update from presenter."""
        self._quality_panel.set_reprojection_data(data)
        self._update_filter_preview()

    def _on_volumetric_accuracy_updated(self, report: VolumetricScaleReport) -> None:
        """Handle volumetric scale accuracy update from presenter."""
        self._latest_scale_report = report
        self._sparkline.set_data(report)
        self._quality_panel.set_volumetric_accuracy(report)

        # Update detail dialog if it exists
        if self._scale_detail_dialog is not None:
            self._scale_detail_dialog.set_data(report)

        # Show/hide expand button based on whether we have data
        if report.n_frames_sampled > 0:
            self._sparkline_expand_btn.show()
        else:
            self._sparkline_expand_btn.hide()

    def _on_sparkline_frame_clicked(self, sync_index: int) -> None:
        """Handle click-to-seek from sparkline."""
        # Map sync_index to slider position
        if len(self._valid_sync_indices) == 0:
            return

        position = int(np.searchsorted(self._valid_sync_indices, sync_index))
        position = min(position, len(self._valid_sync_indices) - 1)
        self._frame_slider.setValue(position)

    def _show_scale_detail_dialog(self) -> None:
        """Show expanded scale accuracy chart."""
        if self._scale_detail_dialog is None:
            self._scale_detail_dialog = ScaleDetailDialog(self)
            self._scale_detail_dialog.frame_clicked.connect(self._on_sparkline_frame_clicked)

        # Populate with current data and cursor position
        if self._latest_scale_report is not None:
            self._scale_detail_dialog.set_data(self._latest_scale_report)
        if len(self._valid_sync_indices) > 0:
            actual_sync_index = int(self._valid_sync_indices[self._frame_slider.value()])
            self._scale_detail_dialog.set_cursor(actual_sync_index)

        self._scale_detail_dialog.show()
        self._scale_detail_dialog.raise_()

    def _on_coverage_updated(self, coverage: np.ndarray, labels: list[str]) -> None:
        """Handle coverage data update from presenter."""
        self._coverage_data = (coverage, labels)
        self._coverage_btn.setEnabled(True)

    def _show_coverage_dialog(self) -> None:
        """Show floating dialog with coverage heatmap."""
        if self._coverage_data is None:
            return

        coverage, labels = self._coverage_data

        dialog = QDialog(self)
        dialog.setWindowTitle("Camera Coverage")
        dialog.setModal(False)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)

        heatmap = CoverageHeatmapWidget()
        heatmap.set_data(coverage, killed_linkages=set(), labels=labels)
        layout.addWidget(heatmap)

        dialog.adjustSize()
        dialog.show()

    def _on_view_model_updated(self, view_model: PlaybackViewModel) -> None:
        """Handle view model update from presenter."""
        if self._pyvista_widget is None:
            if self.isVisible():
                self._create_pyvista_widget(view_model)
            else:
                logger.debug("Deferring PyVista widget creation until tab is visible")
                self._pending_view_model = view_model
        else:
            self._pyvista_widget.set_view_model(view_model, preserve_camera=True)

        self._update_slider_for_view_model(view_model)

    def _create_pyvista_widget(self, view_model: PlaybackViewModel) -> None:
        """Create the PyVista widget. Only call when visible."""
        logger.debug("Creating PyVista widget (tab is visible)")

        self._viz_placeholder.hide()

        self._pyvista_widget = PlaybackVizWidget(view_model)
        self._pyvista_widget.show_playback_controls(False)
        self._viz_layout.insertWidget(0, self._pyvista_widget)
        self._viz_layout.setStretchFactor(self._pyvista_widget, 1)

        # Give 3D widget more splitter space
        self._splitter.setSizes([750, 250])
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 1)

        self._pending_view_model = None

    def _update_slider_for_view_model(self, view_model: PlaybackViewModel) -> None:
        """Update slider state for the given view model."""
        self._valid_sync_indices = view_model.valid_sync_indices
        n_frames = len(self._valid_sync_indices)

        self._frame_slider.blockSignals(True)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(max(0, n_frames - 1))

        if n_frames > 0:
            current_sync = self._presenter.current_sync_index
            position = int(np.searchsorted(self._valid_sync_indices, current_sync))
            position = min(position, n_frames - 1)
            self._frame_slider.setValue(position)

        self._frame_slider.setEnabled(n_frames > 1)
        self._frame_slider.blockSignals(False)

        # Explicitly update sparkline cursor (slider uses blockSignals, so valueChanged won't fire)
        if n_frames > 0:
            actual_sync_index = int(self._valid_sync_indices[self._frame_slider.value()])
            self._sparkline.set_cursor(actual_sync_index)
            if self._scale_detail_dialog is not None:
                self._scale_detail_dialog.set_cursor(actual_sync_index)

        self._update_frame_display()

    def _update_frame_display(self) -> None:
        """Update the frame counter display."""
        if len(self._valid_sync_indices) == 0:
            self._frame_display.setText("0 / 0")
            return

        slider_position = self._frame_slider.value()
        actual_sync_index = int(self._valid_sync_indices[slider_position])
        max_sync_index = int(self._valid_sync_indices[-1])
        self._frame_display.setText(f"{actual_sync_index} / {max_sync_index}")

    # -------------------------------------------------------------------------
    # VTK Lifecycle
    # -------------------------------------------------------------------------

    def showEvent(self, event) -> None:
        """Handle show event - create deferred PyVista widget if needed."""
        super().showEvent(event)

        if self._pending_view_model is not None and self._pyvista_widget is None:
            logger.debug("Tab now visible - creating deferred PyVista widget")
            self._create_pyvista_widget(self._pending_view_model)

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
