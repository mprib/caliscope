"""View for extrinsic calibration workflow.

Two-pane layout: a 3D visualization + playback controls on the left, and a
scrollable workflow strip / calibrate controls / quality tabs / origin section /
filter row on the right. Follows the MVP pattern with state-driven UI updates.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.scale_accuracy import VolumetricScaleReport
from caliscope.core.workflow_status import StepStatus
from caliscope.gui.presenters.extrinsic_calibration_presenter import (
    CalibrationQualityData,
    CalibrationStepData,
    ExtrinsicCalibrationPresenter,
    ExtrinsicCalibrationState,
    OriginOption,
)
from caliscope.gui.theme import Colors, Styles, Typography
from caliscope.gui.view_models.playback_view_model import PlaybackViewModel
from caliscope.gui.widgets.calibration_quality_tabs import CalibrationQualityTabs
from caliscope.gui.widgets.calibration_step_strip import CalibrationStepStrip
from caliscope.gui.widgets.coverage_heatmap import CoverageHeatmapWidget
from caliscope.gui.widgets.distance_sparkline import DistanceSparkline
from caliscope.gui.widgets.lens_model_dialog import LensModelDialog
from caliscope.gui.widgets.qt3d_playback_widget import Qt3DPlaybackWidget
from caliscope.gui.widgets.scale_detail_dialog import ScaleDetailDialog

logger = logging.getLogger(__name__)


class ExtrinsicCalibrationView(QWidget):
    """View for extrinsic calibration workflow.

    Left pane: 3D visualization, frame slider, distance-error sparkline.
    Right pane (scrollable): workflow strip, calibrate section, quality tabs,
    origin section, filter row.

    UI state is derived from presenter state via _update_ui_for_state().
    This prevents state/UI divergence - there's no stored UI state.
    """

    navigation_requested = Signal(str)  # Tab name, bubbled up from the workflow strip

    def __init__(
        self,
        presenter: ExtrinsicCalibrationPresenter,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._presenter = presenter

        self._viz_widget: Qt3DPlaybackWidget | None = None

        # Valid sync indices for frame navigation (sparse data support)
        # Slider position = index into this array, not the actual sync_index
        self._valid_sync_indices: np.ndarray = np.array([], dtype=np.int64)

        # Scale accuracy state
        self._scale_detail_dialog: ScaleDetailDialog | None = None
        self._latest_scale_report: VolumetricScaleReport | None = None

        # Quality state (retained for lens-model/coverage dialog launches)
        self._quality_data: CalibrationQualityData | None = None
        self._coverage_data: tuple[np.ndarray, list[str]] | None = None

        self._setup_ui()
        self._connect_signals()
        self._update_ui_for_state(presenter.state)

        # Restored sessions already have a capture volume but never emit
        # capture_volume_changed (that signal only fires on new operations),
        # so populate the origin combo directly from current state.
        if self._presenter.capture_volume is not None:
            self._populate_origin_combo()

        # Emit initial state (coverage, workflow strip, and if restored session: quality + 3D viz)
        self._presenter.emit_initial_state()

        logger.info("ExtrinsicCalibrationView created")

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the two-pane splitter layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_pane = self._create_left_pane()
        right_pane = self._create_right_pane()

        splitter.addWidget(left_pane)
        splitter.addWidget(right_pane)
        splitter.setStretchFactor(0, 62)
        splitter.setStretchFactor(1, 38)
        splitter.setSizes([620, 380])
        left_pane.setMinimumWidth(300)
        right_pane.setMinimumWidth(320)

        self._splitter = splitter

    def _create_left_pane(self) -> QWidget:
        """3D view + frame slider + distance sparkline."""
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(0)
        self._viz_layout = viz_layout

        # Placeholder shown until visualization widget is created
        self._viz_placeholder = QLabel("Run calibration to see 3D visualization")
        self._viz_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._viz_placeholder.setStyleSheet(
            f"QLabel {{ color: {Colors.TEXT_MUTED}; font-size: 14px; "
            f"background-color: {Colors.SURFACE}; border: 1px dashed {Colors.BORDER}; border-radius: 4px; }}"
        )
        self._viz_placeholder.setFixedHeight(150)
        viz_layout.addWidget(self._viz_placeholder)
        layout.addWidget(viz_container, stretch=1)

        self._playback_controls = self._create_playback_controls()
        layout.addWidget(self._playback_controls)
        self._playback_controls.hide()  # Hidden until a capture volume exists

        return pane

    def _create_playback_controls(self) -> QWidget:
        """Frame slider + distance sparkline, aligned in a shared grid column.

        No leading labels - the sparkline's cursor must track the slider handle,
        so both rows share identical margins in the same grid column.
        """
        group = QWidget()
        grid = QGridLayout(group)
        grid.setContentsMargins(10, 4, 4, 4)  # ~slider handle half-width (10px) on the left
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.setEnabled(False)
        self._frame_slider.setStyleSheet(Styles.SLIDER)
        grid.addWidget(self._frame_slider, 0, 0)

        self._frame_display = QLabel("0 / 0")
        self._frame_display.setMinimumWidth(70)
        grid.addWidget(self._frame_display, 0, 1)

        self._sparkline = DistanceSparkline()
        grid.addWidget(self._sparkline, 1, 0)

        self._sparkline_expand_btn = QToolButton()
        self._sparkline_expand_btn.setText("++")
        self._sparkline_expand_btn.setFixedSize(20, 20)
        self._sparkline_expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sparkline_expand_btn.setToolTip("Show detailed distance error chart")
        self._sparkline_expand_btn.hide()
        grid.addWidget(self._sparkline_expand_btn, 1, 1)

        return group

    def _create_right_pane(self) -> QScrollArea:
        """Scrollable: workflow strip, calibrate section, quality tabs, origin, filter."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # ① Workflow strip - always visible
        self._step_strip = CalibrationStepStrip()
        layout.addWidget(self._step_strip)

        # ② Calibrate section - always visible
        self._calibrate_section = self._create_calibrate_section()
        layout.addWidget(self._calibrate_section)

        # ③④⑤ Quality tabs, origin section, filter row - hidden until a capture volume exists
        self._post_cal_group = QWidget()
        post_cal_layout = QVBoxLayout(self._post_cal_group)
        post_cal_layout.setContentsMargins(0, 0, 0, 0)
        post_cal_layout.setSpacing(12)

        self._quality_tabs = CalibrationQualityTabs()
        post_cal_layout.addWidget(self._quality_tabs)

        self._origin_section = self._create_origin_section()
        post_cal_layout.addWidget(self._origin_section)

        self._filter_row_widget = self._create_filter_row()
        post_cal_layout.addWidget(self._filter_row_widget)

        layout.addWidget(self._post_cal_group)
        self._post_cal_group.hide()

        layout.addStretch()

        scroll.setWidget(content)
        return scroll

    def _create_calibrate_section(self) -> QWidget:
        """Refine-intrinsics checkbox + Calibrate/Recalibrate/Cancel button + progress."""
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._refine_checkbox = QCheckBox("Refine intrinsics during calibration")
        tooltip = (
            "When checked, focal length and distortion are re-estimated jointly with camera "
            "poses. When unchecked, provided intrinsics are locked. Cameras with no intrinsic "
            "calibration always start from an automatic estimate and are always refined."
        )
        intrinsics_available = self._presenter.intrinsics_available
        if not intrinsics_available:
            tooltip += " (required: one or more cameras have no intrinsics)"
        self._refine_checkbox.setToolTip(tooltip)
        self._refine_checkbox.setChecked(self._presenter.refine_intrinsics)
        self._refine_checkbox.setEnabled(intrinsics_available)
        layout.addWidget(self._refine_checkbox)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)

        self._action_btn = QPushButton("Calibrate")
        self._action_btn.setStyleSheet(Styles.PRIMARY_BUTTON)
        action_row.addWidget(self._action_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.hide()
        action_row.addWidget(self._progress_bar, stretch=1)

        layout.addLayout(action_row)

        self._progress_label = QLabel("")
        self._progress_label.hide()
        layout.addWidget(self._progress_label)

        return group

    def _create_origin_section(self) -> QWidget:
        """Origin marker selection, current-origin label, and rotation controls."""
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel("Origin")
        header.setStyleSheet(Typography.SECTION_HEADER)
        layout.addWidget(header)

        combo_row = QHBoxLayout()
        combo_row.setContentsMargins(0, 0, 0, 0)
        combo_row.setSpacing(8)
        self._origin_combo = QComboBox()
        combo_row.addWidget(self._origin_combo, stretch=1)
        self._set_origin_btn = QPushButton("Set Origin")
        self._set_origin_btn.setStyleSheet(Styles.GHOST_BUTTON)
        combo_row.addWidget(self._set_origin_btn)
        layout.addLayout(combo_row)

        self._origin_hint = QLabel()
        self._origin_hint.setStyleSheet(Typography.HELPER_TEXT)
        self._origin_hint.hide()
        layout.addWidget(self._origin_hint)

        self._origin_label = QLabel("Origin: not set")
        self._origin_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        layout.addWidget(self._origin_label)

        rotate_row = QHBoxLayout()
        rotate_row.setContentsMargins(0, 0, 0, 0)
        rotate_row.setSpacing(16)
        rotate_row.addWidget(QLabel("Adjust:"))
        rotate_row.addWidget(self._create_rotation_grid())
        rotate_row.addStretch()
        layout.addLayout(rotate_row)

        return group

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

    def _create_filter_row(self) -> QWidget:
        """Create filter controls row (at bottom)."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

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
        self._presenter.workflow_updated.connect(self._on_workflow_updated)
        self._presenter.coverage_updated.connect(self._on_coverage_updated)
        self._presenter.view_model_updated.connect(self._on_view_model_updated)
        self._presenter.capture_volume_changed.connect(self._on_capture_volume_changed)

        # View -> Presenter
        self._action_btn.clicked.connect(self._on_action_clicked)
        self._refine_checkbox.toggled.connect(self._presenter.set_refine_intrinsics)
        self._filter_apply_btn.clicked.connect(self._on_filter_apply_clicked)
        self._filter_mode.currentIndexChanged.connect(self._on_filter_mode_changed)
        self._filter_value.valueChanged.connect(self._on_filter_value_changed)
        self._set_origin_btn.clicked.connect(self._on_set_origin_clicked)
        self._origin_combo.currentIndexChanged.connect(self._update_origin_hint)
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)

        # Sparkline / detail dialog (position domain - no sync_index translation needed)
        self._sparkline.frame_clicked.connect(self._on_sparkline_frame_clicked)
        self._sparkline_expand_btn.clicked.connect(self._show_scale_detail_dialog)

        # Quality tabs
        self._quality_tabs.view_lens_model_requested.connect(self._show_lens_model_dialog)
        self._quality_tabs.view_coverage_requested.connect(self._show_coverage_dialog)

        # Workflow strip navigation bubbles up to whoever owns this view
        self._step_strip.navigation_requested.connect(self.navigation_requested)

    # -------------------------------------------------------------------------
    # State-Driven UI
    # -------------------------------------------------------------------------

    def _update_ui_for_state(self, state: ExtrinsicCalibrationState) -> None:
        """Update all UI elements based on presenter state."""
        is_running = state == ExtrinsicCalibrationState.CALIBRATING
        has_capture_volume = state == ExtrinsicCalibrationState.CALIBRATED

        # During filter re-optimization the capture volume is preserved (not cleared like a
        # fresh recalibration), so controls stay visible instead of collapsing back down.
        show_calibrated = has_capture_volume or (is_running and self._presenter.capture_volume is not None)

        if is_running:
            self._action_btn.setText("Cancel")
        elif has_capture_volume:
            self._action_btn.setText("Recalibrate")
        else:
            self._action_btn.setText("Calibrate")
        self._action_btn.setEnabled(True)

        self._progress_bar.setVisible(is_running)
        self._progress_label.setVisible(is_running)
        self._refine_checkbox.setEnabled(self._presenter.intrinsics_available and not is_running)

        self._playback_controls.setVisible(show_calibrated)
        self._post_cal_group.setVisible(show_calibrated)

        if show_calibrated:
            # Filter re-optimization: keep last-good numbers visible but greyed out.
            self._quality_tabs.set_disabled_with_last_values(is_running)
            self._origin_section.setEnabled(not is_running)
            self._filter_row_widget.setEnabled(not is_running)
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
        opt: OriginOption | None = self._origin_combo.currentData()
        if opt is None:
            return

        if opt.is_static:
            self._presenter.align_to_origin(opt.object_id, None)
            return

        if len(self._valid_sync_indices) == 0:
            return
        actual_sync_index = int(self._valid_sync_indices[self._frame_slider.value()])
        self._presenter.align_to_origin(opt.object_id, actual_sync_index)

    def _populate_origin_combo(self) -> None:
        """Populate the origin combo from the presenter, preselecting the persisted origin."""
        options = self._presenter.get_origin_options()

        self._origin_combo.blockSignals(True)
        self._origin_combo.clear()
        for opt in options:
            self._origin_combo.addItem(opt.label, userData=opt)

        origin_id = self._presenter.current_origin_object_id
        if origin_id is not None:
            for i in range(self._origin_combo.count()):
                item_opt: OriginOption | None = self._origin_combo.itemData(i)
                if item_opt is not None and item_opt.object_id == origin_id:
                    self._origin_combo.setCurrentIndex(i)
                    break
        self._origin_combo.blockSignals(False)

        self._update_origin_hint()

    def _update_origin_hint(self) -> None:
        """Disable Set Origin and show a hint when the selected moving marker isn't visible here."""
        opt: OriginOption | None = self._origin_combo.currentData()
        if opt is None:
            self._set_origin_btn.setEnabled(False)
            self._origin_hint.hide()
            return

        if opt.is_static or len(self._valid_sync_indices) == 0:
            self._set_origin_btn.setEnabled(True)
            self._origin_hint.hide()
            return

        actual_sync_index = int(self._valid_sync_indices[self._frame_slider.value()])
        visible = self._presenter.is_object_visible_at(opt.object_id, actual_sync_index)
        self._set_origin_btn.setEnabled(visible)
        if visible:
            self._origin_hint.hide()
        else:
            self._origin_hint.setText(f"{opt.label} not visible in this frame")
            self._origin_hint.show()

    def _on_frame_slider_changed(self, position: int) -> None:
        """Handle frame slider value change."""
        if len(self._valid_sync_indices) == 0:
            return

        actual_sync_index = int(self._valid_sync_indices[position])

        if self._viz_widget is not None:
            self._viz_widget.set_sync_index(actual_sync_index)
        self._presenter.set_sync_index(actual_sync_index)
        self._sparkline.set_cursor(position)
        if self._scale_detail_dialog is not None:
            self._scale_detail_dialog.set_cursor(position)
        self._update_frame_display()
        self._update_origin_hint()

    def _on_progress_updated(self, percent: int, message: str) -> None:
        """Handle progress update from presenter."""
        self._progress_bar.setValue(percent)
        self._progress_label.setText(message)

    def _on_quality_updated(self, data: CalibrationQualityData) -> None:
        """Handle quality data update from presenter."""
        self._quality_data = data
        self._quality_tabs.set_data(data)
        self._update_filter_preview()

    def _on_workflow_updated(self, data: CalibrationStepData) -> None:
        """Handle workflow step data update from presenter."""
        self._step_strip.set_data(data)

        origin_status, origin_detail = data.origin
        if origin_status == StepStatus.COMPLETE:
            self._origin_label.setText(f"Origin: {origin_detail}")
        else:
            self._origin_label.setText("Origin: not set")

    def _on_volumetric_accuracy_updated(self, report: VolumetricScaleReport) -> None:
        """Handle volumetric scale accuracy update from presenter."""
        self._latest_scale_report = report
        self._sparkline.set_data(report, self._valid_sync_indices)

        if self._scale_detail_dialog is not None:
            self._scale_detail_dialog.set_data(report, self._valid_sync_indices)

        self._sparkline_expand_btn.setVisible(report.n_frames_sampled > 0)

    def _on_sparkline_frame_clicked(self, position: int) -> None:
        """Handle click-to-seek from the sparkline or detail dialog (slider position domain)."""
        self._frame_slider.setValue(position)

    def _show_scale_detail_dialog(self) -> None:
        """Show expanded distance error chart."""
        if self._scale_detail_dialog is None:
            self._scale_detail_dialog = ScaleDetailDialog(self)
            self._scale_detail_dialog.frame_clicked.connect(self._on_sparkline_frame_clicked)

        if self._latest_scale_report is not None:
            self._scale_detail_dialog.set_data(self._latest_scale_report, self._valid_sync_indices)
        self._scale_detail_dialog.set_cursor(self._frame_slider.value())

        self._scale_detail_dialog.show()
        self._scale_detail_dialog.raise_()

    def _show_lens_model_dialog(self, cam_id: int) -> None:
        """Show lens model visualization for a specific camera."""
        if self._quality_data is None:
            return

        depth_ratios = {row.cam_id: row.depth_ratio for row in self._quality_data.camera_rows}
        dialog = LensModelDialog(
            cameras=self._quality_data.cameras,
            extrinsic_dir=self._quality_data.extrinsic_dir,
            depth_ratios=depth_ratios,
            initial_cam_id=cam_id,
            parent=self,
        )
        dialog.show()

    def _on_coverage_updated(self, coverage: np.ndarray, labels: list[str]) -> None:
        """Handle coverage data update from presenter."""
        self._coverage_data = (coverage, labels)

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

    def _on_capture_volume_changed(self, capture_volume: CaptureVolume) -> None:  # noqa: ARG002
        """Refresh origin combo whenever the capture volume changes (calibrate, filter, rotate, align)."""
        self._populate_origin_combo()

    def _on_view_model_updated(self, view_model: PlaybackViewModel) -> None:
        """Handle view model update from presenter."""
        if self._viz_widget is None:
            self._create_viz_widget(view_model)
        else:
            self._viz_widget.set_view_model(view_model, preserve_camera=True)

        self._update_slider_for_view_model(view_model)

    def _create_viz_widget(self, view_model: PlaybackViewModel) -> None:
        """Create the Qt3D visualization widget."""
        logger.debug("Creating Qt3D visualization widget")

        self._viz_placeholder.hide()

        self._viz_widget = Qt3DPlaybackWidget(
            view_model,
            camera_size_multiplier=self._presenter.get_camera_size_multiplier(),
            grid_size_multiplier=self._presenter.get_grid_size_multiplier(),
            sphere_size_multiplier=self._presenter.get_sphere_size_multiplier(),
        )
        self._viz_widget.camera_size_multiplier_changed.connect(self._presenter.save_camera_size_multiplier)
        self._viz_widget.grid_size_multiplier_changed.connect(self._presenter.save_grid_size_multiplier)
        self._viz_widget.sphere_size_multiplier_changed.connect(self._presenter.save_sphere_size_multiplier)
        self._viz_widget.show_playback_controls(False)
        self._viz_layout.insertWidget(0, self._viz_widget)
        self._viz_layout.setStretchFactor(self._viz_widget, 1)

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

        # Explicitly update cursors (slider uses blockSignals, so valueChanged won't fire)
        if n_frames > 0:
            position = self._frame_slider.value()
            self._sparkline.set_cursor(position)
            if self._scale_detail_dialog is not None:
                self._scale_detail_dialog.set_cursor(position)

        self._update_frame_display()
        self._update_origin_hint()

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
    # Lifecycle
    # -------------------------------------------------------------------------

    def suspend_rendering(self) -> None:
        """Pause 3D rendering when tab not active."""
        if self._viz_widget is not None:
            self._viz_widget.suspend_rendering()

    def resume_rendering(self) -> None:
        """Resume 3D rendering when tab becomes active."""
        if self._viz_widget is not None:
            self._viz_widget.resume_rendering()

    def cleanup(self) -> None:
        """Explicit cleanup - call before destruction."""
        if self._viz_widget is not None:
            self._viz_widget.close()
            self._viz_widget = None
        logger.info("ExtrinsicCalibrationView cleaned up")

    def closeEvent(self, event) -> None:
        """Handle close event."""
        self.cleanup()
        super().closeEvent(event)
