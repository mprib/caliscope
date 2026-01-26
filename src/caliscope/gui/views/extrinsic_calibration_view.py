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

from caliscope.gui.presenters.extrinsic_calibration_presenter import (
    ExtrinsicCalibrationPresenter,
    ExtrinsicCalibrationState,
    QualityPanelData,
)
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

        logger.info("ExtrinsicCalibrationView created")

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Vertical splitter: 3D view on top, controls below
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)

        # Top: 3D visualization container
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        self._viz_layout = viz_layout  # Store for lazy PyVista widget insertion
        splitter.addWidget(viz_container)

        # Bottom: controls panel
        controls_panel = self._create_controls_panel()
        splitter.addWidget(controls_panel)

        # Set splitter sizes (visualization gets 70%, controls 30%)
        splitter.setSizes([700, 300])
        splitter.setStretchFactor(0, 3)  # Visualization stretches more
        splitter.setStretchFactor(1, 1)

    def _create_controls_panel(self) -> QWidget:
        """Create the bottom controls panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Row 1: Action buttons + progress
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)

        self._calibrate_btn = QPushButton("Calibrate")
        self._calibrate_btn.setMinimumWidth(100)
        action_layout.addWidget(self._calibrate_btn)

        self._reoptimize_btn = QPushButton("Re-optimize")
        self._reoptimize_btn.setMinimumWidth(100)
        action_layout.addWidget(self._reoptimize_btn)

        action_layout.addStretch()

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setMinimumWidth(200)
        self._progress_bar.hide()
        action_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.hide()
        action_layout.addWidget(self._progress_label)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.hide()
        action_layout.addWidget(self._cancel_btn)

        layout.addLayout(action_layout)

        # Row 2: Filter controls
        self._filter_group = QGroupBox("Filter")
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

        filter_layout.addSpacing(20)

        self._filter_preview_label = QLabel("Removes observations > ?.?? px")
        self._filter_preview_label.setStyleSheet("color: #888; font-style: italic;")
        filter_layout.addWidget(self._filter_preview_label)

        filter_layout.addStretch()

        layout.addWidget(self._filter_group)

        # Row 3: Coordinate frame controls
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

        # Row 4: Frame slider
        frame_layout = QHBoxLayout()

        frame_layout.addWidget(QLabel("Frame:"))

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.setEnabled(False)
        frame_layout.addWidget(self._frame_slider, stretch=1)

        self._frame_display = QLabel("0 / 0")
        self._frame_display.setMinimumWidth(80)
        frame_layout.addWidget(self._frame_display)

        layout.addLayout(frame_layout)

        # Row 5: Quality panel
        self._quality_panel = QualityPanel()
        layout.addWidget(self._quality_panel)

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
        self._presenter.view_model_updated.connect(self._on_view_model_updated)

        # View → Presenter
        self._calibrate_btn.clicked.connect(self._on_calibrate_clicked)
        self._reoptimize_btn.clicked.connect(self._presenter.re_optimize)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        self._filter_apply_btn.clicked.connect(self._on_filter_apply_clicked)
        self._filter_mode.currentIndexChanged.connect(self._on_filter_mode_changed)
        self._filter_value.valueChanged.connect(self._on_filter_value_changed)
        self._set_origin_btn.clicked.connect(self._on_set_origin_clicked)
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)

    # -------------------------------------------------------------------------
    # State-Driven UI
    # -------------------------------------------------------------------------

    def _update_ui_for_state(self, state: ExtrinsicCalibrationState) -> None:
        """Update all UI elements based on presenter state.

        Single handler that derives entire UI from current state.
        Prevents state/UI divergence - there's no stored UI state.
        """
        # Determine capabilities for current state
        can_calibrate = state == ExtrinsicCalibrationState.NEEDS_BOOTSTRAP
        can_reoptimize = state == ExtrinsicCalibrationState.CALIBRATED
        is_running = state == ExtrinsicCalibrationState.OPTIMIZING
        has_bundle = state in (
            ExtrinsicCalibrationState.NEEDS_OPTIMIZATION,
            ExtrinsicCalibrationState.CALIBRATED,
        )

        # Action buttons
        self._calibrate_btn.setEnabled(can_calibrate)
        self._reoptimize_btn.setEnabled(can_reoptimize)

        # Progress and cancel visibility
        self._progress_bar.setVisible(is_running)
        self._progress_label.setVisible(is_running)
        self._cancel_btn.setVisible(is_running)

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

    def _on_calibrate_clicked(self) -> None:
        """Handle calibrate button click."""
        self._presenter.run_calibration()

    def _on_cancel_clicked(self) -> None:
        """Handle cancel button click."""
        self._presenter.cleanup()

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
            # Get threshold for the percentile we're removing
            int_pct = int(value)
            # Find closest available percentile
            if int_pct in preview_data.threshold_at_percentile:
                threshold = preview_data.threshold_at_percentile[int_pct]
                self._filter_preview_label.setText(f"Removes observations > {threshold:.2f} px")
            else:
                # Show mean for rough estimate
                self._filter_preview_label.setText(f"Mean error: {preview_data.mean_error:.2f} px")
        else:
            # Absolute mode - show what we'd remove
            self._filter_preview_label.setText(f"Removes observations > {value:.2f} px")

    def _on_rotate_clicked(self, axis: str, degrees: float) -> None:
        """Handle rotation button click."""
        self._presenter.rotate(axis, degrees)

    def _on_set_origin_clicked(self) -> None:
        """Handle set origin button click."""
        sync_index = self._frame_slider.value()
        self._presenter.align_to_origin(sync_index)

    def _on_frame_slider_changed(self, value: int) -> None:
        """Handle frame slider value change."""
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

    def _on_view_model_updated(self, view_model: PlaybackViewModel) -> None:
        """Handle view model update from presenter."""
        # Create PyVista widget on first use (lazy initialization)
        if self._pyvista_widget is None:
            self._pyvista_widget = PlaybackTriangulationWidgetPyVista(view_model)
            self._pyvista_widget.show_playback_controls(False)  # We have our own slider
            self._viz_layout.addWidget(self._pyvista_widget)
        else:
            self._pyvista_widget.set_view_model(view_model)

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
