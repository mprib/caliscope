"""
Three-panel visualization widget for comparing calibration stages.

Displays ground truth, noisy input, and optimized result side-by-side
with a shared slider for synchronized frame navigation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from caliscope.ui.viz.playback_triangulation_widget_pyvista import PlaybackTriangulationWidgetPyVista
from caliscope.ui.viz.playback_view_model import PlaybackViewModel

if TYPE_CHECKING:
    from tests.synthetic.test_cases import ExtrinsicCalibrationTestCase

# Camera scale for synthetic scenes (mm units, ~4000mm extent)
# Default production scale (0.0002) produces frustums 0.16mm deep - invisible.
# This scale produces frustums ~200mm deep (5% of scene).
SYNTHETIC_CAMERA_SCALE = 0.25


class CalibrationStoryboardWidget(QWidget):
    """
    Three-panel visualization comparing ground truth, noisy input, and optimized result.

    Layout:
    +-------------------+-------------------+-------------------+
    |   GROUND TRUTH    |    NOISY INPUT    |     OPTIMIZED     |
    | [PlaybackWidget]  | [PlaybackWidget]  | [PlaybackWidget]  |
    +-------------------+-------------------+-------------------+
    | Frame: [=========] 20/50                                  |
    | Initial Error: Rot 2.3 deg | Trans 45.2 mm                |
    | Final Error:   Rot 0.1 deg | Trans 1.8 mm                 |
    +-----------------------------------------------------------+
    """

    def __init__(self, test_case: ExtrinsicCalibrationTestCase, parent: QWidget | None = None):
        super().__init__(parent)
        self.test_case = test_case
        self._sync_in_progress = False
        self._setup_ui()
        self._connect_signals()
        self._connect_camera_sync()
        self._update_error_display()

    def _setup_ui(self) -> None:
        """Build the widget layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Create view models for each panel
        self._gt_vm = PlaybackViewModel(
            camera_array=self.test_case.ground_truth.cameras,
            world_points=self.test_case.ground_truth.world_points,
        )
        self._noisy_vm = PlaybackViewModel(
            camera_array=self.test_case.noisy_input.cameras,
            world_points=self._triangulate_from_noisy(),
        )
        self._opt_vm = PlaybackViewModel(
            camera_array=self.test_case.optimized_bundle.camera_array,
            world_points=self.test_case.optimized_bundle.world_points,
        )

        # Three-panel layout
        panels_layout = QHBoxLayout()

        # Ground truth panel
        gt_panel = self._create_panel("GROUND TRUTH", self._gt_vm)
        panels_layout.addWidget(gt_panel, stretch=1)

        # Noisy input panel
        noisy_panel = self._create_panel("NOISY INPUT", self._noisy_vm)
        panels_layout.addWidget(noisy_panel, stretch=1)

        # Optimized panel
        opt_panel = self._create_panel("OPTIMIZED", self._opt_vm)
        panels_layout.addWidget(opt_panel, stretch=1)

        layout.addLayout(panels_layout, stretch=1)

        # Master slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(len(self.test_case.sync_indices) - 1)
        slider_layout.addWidget(self._slider, stretch=1)
        self._frame_label = QLabel("0 / 0")
        slider_layout.addWidget(self._frame_label)
        layout.addLayout(slider_layout)

        # Error display (monospace QTextEdit for aligned columns)
        self._error_display = QTextEdit()
        self._error_display.setReadOnly(True)
        self._error_display.setFont(QFont("Monospace", 10))
        self._error_display.setMaximumHeight(180)
        layout.addWidget(self._error_display)

    def _create_panel(self, title: str, view_model: PlaybackViewModel) -> QWidget:
        """Create a labeled panel with a PlaybackTriangulationWidget."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)

        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(label)

        widget = PlaybackTriangulationWidgetPyVista(
            view_model,
            camera_scale=SYNTHETIC_CAMERA_SCALE,
        )
        widget.show_playback_controls(False)  # Hide individual controls
        layout.addWidget(widget, stretch=1)

        # Store reference for later using sanitized attribute name
        attr_name = f"_{title.lower().replace(' ', '_')}_widget"
        setattr(self, attr_name, widget)

        return panel

    def _triangulate_from_noisy(self):
        """Triangulate world points from noisy observations using perturbed cameras."""
        return self.test_case.noisy_input.image_points.triangulate(self.test_case.noisy_input.cameras)

    def _connect_signals(self) -> None:
        """Connect slider to all panels."""
        self._slider.valueChanged.connect(self._on_slider_changed)

    def _on_slider_changed(self, value: int) -> None:
        """Broadcast sync_index change to all panels."""
        sync_index = self.test_case.sync_indices[value]

        # Update frame label
        self._frame_label.setText(f"{value + 1} / {len(self.test_case.sync_indices)}")

        # Update all panels
        self._ground_truth_widget.set_sync_index(sync_index)
        self._noisy_input_widget.set_sync_index(sync_index)
        self._optimized_widget.set_sync_index(sync_index)

    def _connect_camera_sync(self) -> None:
        """Connect VTK observers for camera synchronization across panels."""
        widgets = [self._ground_truth_widget, self._noisy_input_widget, self._optimized_widget]
        for widget in widgets:
            if hasattr(widget.plotter, "iren") and widget.plotter.iren is not None:
                vtk_iren = widget.plotter.iren.interactor
                if vtk_iren is not None:
                    # Capture widget in closure with default argument
                    vtk_iren.AddObserver(
                        "EndInteractionEvent",
                        lambda obj, event, w=widget: self._sync_camera_from(w),
                    )

    def _sync_camera_from(self, source: PlaybackTriangulationWidgetPyVista) -> None:
        """Copy camera state from source to all other panels."""
        if self._sync_in_progress:
            return

        self._sync_in_progress = True
        try:
            cam = source.plotter.camera
            widgets = [self._ground_truth_widget, self._noisy_input_widget, self._optimized_widget]
            for widget in widgets:
                if widget is not source:
                    widget.plotter.camera.position = cam.position
                    widget.plotter.camera.focal_point = cam.focal_point
                    widget.plotter.camera.up = cam.up
                    widget.plotter.render()
        finally:
            self._sync_in_progress = False

    def _update_error_display(self) -> None:
        """
        Update the error statistics display with RMSE and per-camera breakdown.

        ELI5: Why these numbers matter
        ─────────────────────────────
        We inject fake "noise" into perfect synthetic data, then see if bundle
        adjustment can recover the original camera positions. Think of it like:

        1. We know exactly where the cameras are (ground truth)
        2. We blur the 2D detections a bit (pixel noise, e.g. 0.5 pixels)
        3. We perturb the camera positions (starting error)
        4. Bundle adjustment tries to find the best positions
        5. We measure how close it got to ground truth (final error)

        Theory (from covariance propagation):
        - RMSE should converge to roughly the pixel noise level
        - Translation error scales linearly: ~15-20 mm per pixel of noise
        - For 0.5 px noise, expect ~7-10 mm max translation error

        If RMSE ≈ pixel_noise, the optimizer converged properly.
        If translation error scales linearly with noise, there are no bugs.
        """
        lines = []

        # RMSE section - shows optimizer convergence
        lines.append("REPROJECTION RMSE")
        lines.append(f"  Initial: {self.test_case.initial_reprojection_rmse:6.2f} px")
        lines.append(f"  Final:   {self.test_case.final_reprojection_rmse:6.2f} px")
        lines.append("")

        # Per-camera pose error breakdown (all cameras - no gauge reference needed)
        # Gauge freedom is resolved by align_to_object() using known object coordinates
        lines.append("POSE ERROR BY CAMERA (vs ground truth)")
        for port in sorted(self.test_case.initial_pose_errors.keys()):
            initial = self.test_case.initial_pose_errors[port]
            final = self.test_case.final_pose_errors[port]
            lines.append(
                f"  Cam {port}: Rot {initial.rotation_deg:5.2f} -> {final.rotation_deg:5.2f} deg | "
                f"Trans {initial.translation_mm:6.1f} -> {final.translation_mm:5.1f} mm"
            )

        self._error_display.setText("\n".join(lines))
