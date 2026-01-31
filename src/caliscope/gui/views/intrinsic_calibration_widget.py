"""Intrinsic calibration widget for single-camera calibration workflow.

Provides video playback with charuco tracking overlay, calibration controls,
and results display. Connects to IntrinsicCalibrationPresenter for business logic.
"""

import logging
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event
from typing import Any

import cv2
from numpy.typing import NDArray
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from caliscope.cameras.camera_array import CameraData
from caliscope.core.calibrate_intrinsics import IntrinsicCalibrationOutput
from caliscope.gui.frame_emitters.tools import (
    apply_rotation,
    cv2_to_qlabel,
    resize_to_square,
)
from caliscope.gui.lens_model_visualizer import LensModelVisualizer
from caliscope.gui.presenters.intrinsic_calibration_presenter import (
    IntrinsicCalibrationPresenter,
    IntrinsicCalibrationState,
)
from caliscope.gui.theme import Styles
from caliscope.packets import FramePacket, PointPacket

logger = logging.getLogger(__name__)


@dataclass
class OverlaySettings:
    """User-toggleable overlay visibility."""

    show_current_points: bool = True
    show_accumulated: bool = True
    show_selected_grids: bool = True


class CalibrationResultsDisplay(QWidget):
    """Display intrinsic calibration results.

    Shows camera matrix parameters, distortion coefficients, and fit quality
    metrics. Always visible with placeholder values until calibration populates them.

    Quality color-coding for RMSE values:
    - Green (< 0.5px): Excellent for motion capture
    - Yellow (0.5-1.0px): Acceptable for most applications
    - Red (> 1.0px): May affect 3D reconstruction accuracy
    """

    # RMSE thresholds for quality color-coding
    RMSE_EXCELLENT = 0.5  # pixels
    RMSE_ACCEPTABLE = 1.0  # pixels

    # Style for row labels with tooltip indicator (dotted underline)
    # QToolTip rule ensures tooltip popup doesn't inherit the underline
    _INFO_LABEL_STYLE = (
        "QLabel { text-decoration: underline dotted; text-decoration-color: #888; "
        "text-underline-offset: 2px; } "
        "QToolTip { text-decoration: none; }"
    )

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _create_info_label(self, text: str, tooltip: str) -> QLabel:
        """Create a row label with discoverable tooltip styling (dotted underline)."""
        label = QLabel(text)
        label.setToolTip(tooltip)
        label.setStyleSheet(self._INFO_LABEL_STYLE)
        return label

    def _setup_ui(self) -> None:
        """Create the form layout with labeled value fields."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Fit quality group
        fit_group = QGroupBox("Fit Quality")
        fit_group.setToolTip("How well the calibration model fits the observed data.")
        fit_layout = QFormLayout(fit_group)

        self._rmse_label = QLabel("—")
        rmse_row = self._create_info_label(
            "RMSE:",
            "Root Mean Square Error of reprojection: the average distance (in pixels) "
            "between where calibration predicts a point should appear and where it "
            "was actually detected.\n\n"
            "Under 0.5px = excellent\n"
            "0.5–1.0px = good for most applications\n"
            "Over 1.0px = review your data quality\n\n"
            "Note: Very low error with few frames or poor coverage may indicate "
            "overfitting rather than good calibration.",
        )

        self._grid_count_label = QLabel("—")
        frames_row = self._create_info_label(
            "Frames used:",
            "Number of frames where the calibration board was detected.\n\n"
            "Minimum: 15–20 frames (fewer may be unstable)\n"
            "Recommended: 30–50 frames\n"
            "Diminishing returns beyond ~100 frames\n\n"
            "Quality matters more than quantity — 25 well-distributed frames "
            "beat 100 frames from similar viewpoints.",
        )

        fit_layout.addRow(rmse_row, self._rmse_label)
        fit_layout.addRow(frames_row, self._grid_count_label)
        layout.addWidget(fit_group)

        # Coverage quality group
        coverage_group = QGroupBox("Coverage Quality")
        coverage_group.setToolTip("How well calibration frames sample the camera's field of view.")
        coverage_layout = QFormLayout(coverage_group)

        self._coverage_label = QLabel("—")
        coverage_row = self._create_info_label(
            "Image coverage:",
            "Percentage of the image area where calibration corners were detected.\n\n"
            "Why edges matter: Lens distortion is strongest near image edges — often "
            "10× greater than at center. Without edge measurements, calibration must "
            "extrapolate, reducing accuracy.\n\n"
            "Why edges are hard: The detector needs complete board squares to find "
            "corners. When the board extends past the frame edge, those corners can't "
            "be detected.\n\n"
            "Target: 80%+ coverage with observations near all four edges.",
        )

        self._orientation_label = QLabel("—")
        orientation_row = self._create_info_label(
            "Board orientations:",
            "Count of distinct board tilts captured.\n\n"
            "Why this matters: When the board is tilted, one edge appears closer "
            "(larger) and the opposite edge appears farther (smaller). This "
            "'foreshortening' provides geometric information needed to accurately "
            "determine focal length.\n\n"
            "Capturing the board only flat (parallel to camera) creates ambiguity "
            "that can cause calibration to fail.\n\n"
            "Minimum: 3–4 distinct orientations\n"
            "Better: Tilt in multiple directions (left, right, up, down, diagonal)",
        )

        coverage_layout.addRow(coverage_row, self._coverage_label)
        coverage_layout.addRow(orientation_row, self._orientation_label)
        layout.addWidget(coverage_group)

        # Camera matrix group
        matrix_group = QGroupBox("Camera Matrix")
        matrix_group.setToolTip("Internal properties of your camera and lens, computed during calibration.")
        matrix_layout = QFormLayout(matrix_group)

        self._fx_label = QLabel("—")
        fx_row = self._create_info_label(
            "fx:",
            "Horizontal focal length in pixel units. Larger values mean a more zoomed-in (narrower) field of view.",
        )

        self._fy_label = QLabel("—")
        fy_row = self._create_info_label(
            "fy:",
            "Vertical focal length in pixel units. Almost always matches fx.",
        )

        self._cx_label = QLabel("—")
        cx_row = self._create_info_label(
            "cx:",
            "Horizontal center of the lens in pixels. Ideally near image center, "
            "but manufacturing tolerances mean it's rarely exact.",
        )

        self._cy_label = QLabel("—")
        cy_row = self._create_info_label(
            "cy:",
            "Vertical center of the lens in pixels.",
        )

        matrix_layout.addRow(fx_row, self._fx_label)
        matrix_layout.addRow(fy_row, self._fy_label)
        matrix_layout.addRow(cx_row, self._cx_label)
        matrix_layout.addRow(cy_row, self._cy_label)
        layout.addWidget(matrix_group)

        # Distortion coefficients group
        dist_group = QGroupBox("Distortion")
        dist_group.setToolTip(
            "Corrects for lens distortion where straight lines appear curved, especially near image edges."
        )
        dist_layout = QFormLayout(dist_group)

        self._k1_label = QLabel("—")
        k1_row = self._create_info_label(
            "k1:",
            "Main distortion term. Negative = barrel distortion (edges bow outward), "
            "positive = pincushion (edges bow inward).",
        )

        self._k2_label = QLabel("—")
        k2_row = self._create_info_label(
            "k2:",
            "Secondary distortion term. Improves accuracy for lenses with stronger distortion.",
        )

        self._p1_label = QLabel("—")
        p1_row = self._create_info_label(
            "p1:",
            "Corrects for the lens not being perfectly parallel to the sensor. Usually small.",
        )

        self._p2_label = QLabel("—")
        p2_row = self._create_info_label(
            "p2:",
            "Vertical lens-sensor alignment correction. Usually small.",
        )

        self._k3_label = QLabel("—")
        k3_row = self._create_info_label(
            "k3:",
            "Third radial distortion term. k1 and k2 capture most distortion; k3 "
            "models residual higher-order effects.\n\n"
            "Typically small but non-zero for most lenses. Larger values are common "
            "with wide-angle lenses where distortion is more complex.",
        )

        dist_layout.addRow(k1_row, self._k1_label)
        dist_layout.addRow(k2_row, self._k2_label)
        dist_layout.addRow(p1_row, self._p1_label)
        dist_layout.addRow(p2_row, self._p2_label)
        dist_layout.addRow(k3_row, self._k3_label)
        layout.addWidget(dist_group)

    def _format_rmse_with_color(self, rmse: float) -> str:
        """Format RMSE value with color-coded quality indicator.

        Returns HTML span with color based on quality thresholds.
        """
        if rmse < self.RMSE_EXCELLENT:
            color = "#4CAF50"  # Green - excellent
        elif rmse < self.RMSE_ACCEPTABLE:
            color = "#FFC107"  # Yellow/amber - acceptable
        else:
            color = "#F44336"  # Red - needs attention

        return f'<span style="color: {color}; font-weight: bold;">{rmse:.3f} px</span>'

    def _format_percentage(self, fraction: float) -> str:
        """Format a 0-1 fraction as a percentage string."""
        return f"{fraction * 100:.0f}%"

    def update_from_output(self, output: IntrinsicCalibrationOutput) -> None:
        """Populate display from calibration output.

        Args:
            output: Complete calibration output with camera params and quality report.
        """
        camera = output.camera
        report = output.report

        # Fit quality - RMSE with color-coding
        self._rmse_label.setText(self._format_rmse_with_color(report.rmse))
        self._grid_count_label.setText(str(report.frames_used))

        # Coverage quality
        self._coverage_label.setText(self._format_percentage(report.coverage_fraction))

        # Orientation - show count and sufficiency
        orientation_text = f"{report.orientation_count}"
        if report.orientation_sufficient:
            orientation_text += " (good)"
        else:
            orientation_text += " (need more variety)"
        self._orientation_label.setText(orientation_text)

        # Camera matrix (guard against None)
        if camera.matrix is not None:
            fx = camera.matrix[0, 0]
            fy = camera.matrix[1, 1]
            cx = camera.matrix[0, 2]
            cy = camera.matrix[1, 2]
            self._fx_label.setText(f"{fx:.1f} px")
            self._fy_label.setText(f"{fy:.1f} px")
            self._cx_label.setText(f"{cx:.1f} px")
            self._cy_label.setText(f"{cy:.1f} px")
        else:
            for label in (self._fx_label, self._fy_label, self._cx_label, self._cy_label):
                label.setText("—")

        # Distortion coefficients (guard against None and variable length)
        if camera.distortions is not None and len(camera.distortions) >= 5:
            k1, k2, p1, p2, k3 = camera.distortions[:5]
            self._k1_label.setText(f"{k1:.6f}")
            self._k2_label.setText(f"{k2:.6f}")
            self._p1_label.setText(f"{p1:.6f}")
            self._p2_label.setText(f"{p2:.6f}")
            self._k3_label.setText(f"{k3:.6f}")
        else:
            for label in (self._k1_label, self._k2_label, self._p1_label, self._p2_label, self._k3_label):
                label.setText("—")

    def reset(self) -> None:
        """Reset all values to placeholder state."""
        # Fit quality
        self._rmse_label.setText("—")
        self._grid_count_label.setText("—")

        # Coverage quality
        self._coverage_label.setText("—")
        self._orientation_label.setText("—")

        # Camera matrix
        self._fx_label.setText("—")
        self._fy_label.setText("—")
        self._cx_label.setText("—")
        self._cy_label.setText("—")

        # Distortion
        self._k1_label.setText("—")
        self._k2_label.setText("—")
        self._p1_label.setText("—")
        self._p2_label.setText("—")
        self._k3_label.setText("—")


class FrameRenderThread(QThread):
    """Processes raw frames for display - runs off GUI thread.

    Reads directly from Presenter's display_queue (no intermediate signal).
    Applies display transforms and emits QPixmaps for the GUI thread.
    Handles overlay rendering for point visualization layers.
    """

    pixmap_ready = Signal(QPixmap)

    # Overlay colors (BGR format for OpenCV)
    CURRENT_POINTS_COLOR = (0, 0, 220)  # Red
    ACCUMULATED_COLOR = (128, 128, 0)  # Teal
    SELECTED_GRIDS_COLOR = (255, 200, 0)  # Bright cyan

    def __init__(
        self,
        display_queue: Queue[FramePacket | None],
        camera: CameraData,
        presenter: IntrinsicCalibrationPresenter,
        pixmap_edge_length: int = 500,
        parent: QThread | None = None,
    ):
        super().__init__(parent)
        self._display_queue = display_queue
        self._camera = camera
        self._presenter = presenter
        self._pixmap_edge_length = pixmap_edge_length
        self._undistort_enabled = False
        self._visualizer: LensModelVisualizer | None = None
        self._keep_running = Event()
        self._overlay_settings = OverlaySettings()

        # Cache last packet for re-rendering when overlay settings change
        self._last_packet: FramePacket | None = None

        # Compute overlay sizes based on image dimensions
        width = camera.size[0]
        self._accumulated_radius = max(4, width // 400)
        self._grid_line_thickness = max(2, width // 600)
        self._current_point_radius = max(5, width // 300)
        self._current_point_thickness = max(2, width // 500)

    def set_undistort(self, enabled: bool, calibrated_camera: CameraData | None) -> None:
        """Enable/disable undistortion."""
        self._undistort_enabled = enabled

        # Create visualizer on first enable
        if enabled and self._visualizer is None and calibrated_camera is not None:
            self._visualizer = LensModelVisualizer(calibrated_camera)

    def set_overlay_visibility(
        self,
        current_points: bool,
        accumulated: bool,
        selected_grids: bool,
    ) -> None:
        """Configure which overlay layers to show."""
        self._overlay_settings.show_current_points = current_points
        self._overlay_settings.show_accumulated = accumulated
        self._overlay_settings.show_selected_grids = selected_grids

    @property
    def shows_boundary(self) -> bool:
        """True if the visualizer draws the original frame boundary."""
        if self._visualizer is None or not self._undistort_enabled:
            return False
        return self._visualizer.content_expands_beyond_frame

    def stop(self) -> None:
        """Signal thread to stop."""
        self._keep_running.clear()

    def rerender_cached(self) -> None:
        """Re-render the last packet with current overlay settings.

        Call this when overlay visibility changes instead of requesting
        a new frame from the presenter.
        """
        if self._last_packet is not None:
            self._render_packet(self._last_packet)

    def _draw_current_points(self, frame: NDArray[Any], points: PointPacket) -> NDArray[Any]:
        """Draw current frame's detected points as red circles."""
        for x, y in points.img_loc:
            cv2.circle(
                frame,
                (int(x), int(y)),
                self._current_point_radius,
                self.CURRENT_POINTS_COLOR,
                self._current_point_thickness,
            )
        return frame

    def _draw_accumulated(self, frame: NDArray[Any]) -> NDArray[Any]:
        """Draw accumulated points as semi-transparent teal circles."""
        collected = self._presenter.collected_points
        if not collected:
            return frame

        overlay = frame.copy()
        for _, points in collected:
            for x, y in points.img_loc:
                cv2.circle(
                    overlay,
                    (int(x), int(y)),
                    self._accumulated_radius,
                    self.ACCUMULATED_COLOR,
                    -1,
                )

        return cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    def _draw_selected_grids(self, frame: NDArray[Any]) -> NDArray[Any]:
        """Draw grids for ALL selected calibration frames at once (coverage map)."""
        selected = self._presenter.selected_frame_indices
        if selected is None:
            return frame

        selected_set = set(selected)
        connectivity = self._presenter.board_connectivity

        # Draw grids for all selected frames simultaneously
        for frame_idx, points in self._presenter.collected_points:
            if frame_idx not in selected_set:
                continue

            id_to_loc = {int(pid): (int(x), int(y)) for pid, (x, y) in zip(points.point_id, points.img_loc)}

            for id_a, id_b in connectivity:
                if id_a in id_to_loc and id_b in id_to_loc:
                    cv2.line(
                        frame,
                        id_to_loc[id_a],
                        id_to_loc[id_b],
                        self.SELECTED_GRIDS_COLOR,
                        self._grid_line_thickness,
                    )

        return frame

    def _render_packet(self, packet: FramePacket) -> None:
        """Render a packet with current overlay settings and emit pixmap."""
        if packet.frame is None:
            return

        # Start with raw frame (View owns all rendering)
        frame = packet.frame.copy()

        # Layer 1: Accumulated points (behind current)
        if self._overlay_settings.show_accumulated:
            frame = self._draw_accumulated(frame)

        # Layer 2: Selected board grids (coverage map)
        if self._overlay_settings.show_selected_grids:
            frame = self._draw_selected_grids(frame)

        # Layer 3: Current frame points (on top)
        if self._overlay_settings.show_current_points and packet.points is not None:
            frame = self._draw_current_points(frame, packet.points)

        # Undistortion
        if self._undistort_enabled and self._visualizer is not None:
            frame = self._visualizer.undistort(frame)

        frame = resize_to_square(frame)
        frame = apply_rotation(frame, self._camera.rotation_count)

        image = cv2_to_qlabel(frame)
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self._pixmap_edge_length,
            self._pixmap_edge_length,
            Qt.AspectRatioMode.KeepAspectRatio,
        )

        self.pixmap_ready.emit(pixmap)

    def run(self) -> None:
        """Main render loop - reads directly from Presenter's queue."""
        self._keep_running.set()
        logger.debug(f"Frame render thread started for port {self._camera.port}")

        while self._keep_running.is_set():
            try:
                packet = self._display_queue.get(timeout=0.1)
            except Empty:
                continue

            # None sentinel - Presenter signals end of sequence
            if packet is None:
                continue

            # Skip packets with no frame (e.g., end-of-stream markers)
            if packet.frame is None:
                continue

            # Cache for re-rendering on overlay toggle
            self._last_packet = packet
            self._render_packet(packet)

        logger.debug(f"Frame render thread exiting for port {self._camera.port}")


class IntrinsicCalibrationWidget(QWidget):
    """Minimal development View for testing IntrinsicCalibrationPresenter.

    Layout:
    - Frame display (QLabel)
    - Frame jogger slider
    - Calibrate button and Undistort checkbox
    """

    def __init__(
        self,
        presenter: IntrinsicCalibrationPresenter,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._presenter = presenter
        self._user_dragging = False

        self._setup_ui()
        self._setup_render_thread()
        self._connect_signals()

        # Initial UI state
        self._update_ui_for_state(presenter.state)

        # Handle restored calibration state (from session cache)
        if presenter.state == IntrinsicCalibrationState.CALIBRATED:
            self._restore_calibrated_state()

    def _setup_ui(self) -> None:
        """Create UI elements."""
        # Main horizontal layout: results on left, video+controls on right
        main_layout = QHBoxLayout(self)

        # Left column: Calibration results (always visible)
        self._results_display = CalibrationResultsDisplay()
        main_layout.addWidget(self._results_display)

        # Right column: Video display and controls, vertically centered
        right_column = QVBoxLayout()
        right_column.setContentsMargins(0, 0, 0, 0)

        # Add stretch to push video unit toward center
        right_column.addStretch(1)

        # Video controls container - groups video, slider, and buttons tightly
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(6)  # Tight spacing between elements

        # Frame display
        self._frame_label = QLabel()
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumSize(500, 500)
        self._frame_label.setStyleSheet("background-color: #1a1a1a;")
        video_layout.addWidget(self._frame_label)

        # Legend for boundary overlay (hidden by default)
        self._boundary_legend = QLabel("Original frame boundary")
        self._boundary_legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._boundary_legend.setStyleSheet("color: #00FFFF;")  # Cyan to match boundary
        self._boundary_legend.hide()
        video_layout.addWidget(self._boundary_legend)

        # Position slider row - directly beneath video
        slider_row = QHBoxLayout()
        slider_row.setSpacing(8)

        self._position_slider = QSlider(Qt.Orientation.Horizontal)
        self._position_slider.setMinimum(0)
        self._position_slider.setMaximum(max(0, self._presenter.frame_count - 1))
        self._position_slider.setStyleSheet(Styles.SLIDER)
        slider_row.addWidget(self._position_slider)

        self._frame_counter = QLabel(f"0 / {self._presenter.frame_count - 1}")
        self._frame_counter.setMinimumWidth(100)
        slider_row.addWidget(self._frame_counter)

        video_layout.addLayout(slider_row)

        # Controls row - Calibrate button and Undistort checkbox, centered together
        controls = QHBoxLayout()
        controls.setSpacing(16)  # Space between button and checkbox

        controls.addStretch()  # Push controls to center

        self._calibrate_btn = QPushButton("Calibrate")
        self._calibrate_btn.setStyleSheet(Styles.PRIMARY_BUTTON)
        self._calibrate_btn.clicked.connect(self._on_calibrate_clicked)
        controls.addWidget(self._calibrate_btn)

        self._undistort_checkbox = QCheckBox("Undistort")
        self._undistort_checkbox.setEnabled(False)
        self._undistort_checkbox.toggled.connect(self._on_undistort_toggled)
        controls.addWidget(self._undistort_checkbox)

        controls.addStretch()  # Balance - push controls to center

        video_layout.addLayout(controls)

        # Add the video container to right column
        right_column.addWidget(video_container)

        # Add stretch below to complete vertical centering
        right_column.addStretch(1)

        # TODO: overlay checkboxes removed as dead code - revisit if needed
        # Previously had: Current Points, All Points, Selected Grids checkboxes
        # The FrameRenderThread still supports overlay rendering if these are restored

        main_layout.addLayout(right_column)

    def _setup_render_thread(self) -> None:
        """Create and start the frame render thread."""
        self._render_thread = FrameRenderThread(
            display_queue=self._presenter.display_queue,
            camera=self._presenter.camera,
            presenter=self._presenter,
        )
        self._render_thread.pixmap_ready.connect(self._on_pixmap_ready)
        self._render_thread.start()

    def _connect_signals(self) -> None:
        """Connect presenter signals to view slots."""
        self._presenter.state_changed.connect(self._update_ui_for_state)
        self._presenter.calibration_complete.connect(self._on_calibration_complete)
        self._presenter.calibration_failed.connect(self._on_calibration_failed)

        # Slider user interaction tracking
        self._position_slider.sliderPressed.connect(self._on_slider_pressed)
        self._position_slider.sliderReleased.connect(self._on_slider_released)
        self._position_slider.valueChanged.connect(self._on_slider_changed)

        # Position tracking from presenter
        self._presenter.frame_position_changed.connect(self._on_position_changed)

    def _on_pixmap_ready(self, pixmap: QPixmap) -> None:
        """Update frame display."""
        self._frame_label.setPixmap(pixmap)

    def _on_slider_pressed(self) -> None:
        """User started dragging slider."""
        self._user_dragging = True

    def _on_slider_released(self) -> None:
        """User released slider."""
        self._user_dragging = False

    def _on_slider_changed(self, value: int) -> None:
        """Slider value changed - seek only if user is dragging."""
        if self._user_dragging and self._position_slider.isEnabled():
            self._presenter.seek_to(value)

    def _on_position_changed(self, frame_index: int) -> None:
        """Presenter reports position - update slider and counter."""
        self._position_slider.blockSignals(True)
        try:
            self._position_slider.setValue(frame_index)
        finally:
            self._position_slider.blockSignals(False)

        self._frame_counter.setText(f"{frame_index} / {self._presenter.frame_count - 1}")

    def _update_ui_for_state(self, state: IntrinsicCalibrationState) -> None:
        """Update UI elements based on presenter state."""
        if state == IntrinsicCalibrationState.READY:
            self._calibrate_btn.setText("Calibrate")
            self._calibrate_btn.setEnabled(True)
            self._undistort_checkbox.setEnabled(False)
            self._position_slider.setEnabled(True)
        elif state == IntrinsicCalibrationState.COLLECTING:
            self._calibrate_btn.setText("Stop")
            self._calibrate_btn.setEnabled(True)
            self._undistort_checkbox.setEnabled(False)
            self._position_slider.setEnabled(False)
        elif state == IntrinsicCalibrationState.CALIBRATING:
            self._calibrate_btn.setText("Calibrating...")
            self._calibrate_btn.setEnabled(False)
            self._undistort_checkbox.setEnabled(False)
            self._position_slider.setEnabled(False)
        elif state == IntrinsicCalibrationState.CALIBRATED:
            self._calibrate_btn.setText("Recalibrate")
            self._calibrate_btn.setEnabled(True)
            self._undistort_checkbox.setEnabled(True)
            self._position_slider.setEnabled(True)

    def _restore_calibrated_state(self) -> None:
        """Initialize display for restored calibration state (from session cache).

        Called on widget init when presenter starts in CALIBRATED state due to
        restoring a previous calibration's report and collected points.
        """
        report = self._presenter.calibration_report
        camera = self._presenter.calibrated_camera
        if report is None or camera is None:
            return

        output = IntrinsicCalibrationOutput(camera=camera, report=report)
        self._results_display.update_from_output(output)

        # Auto-enable undistort to show calibration effect
        self._undistort_checkbox.setChecked(True)

    def _on_calibrate_clicked(self) -> None:
        """Handle calibrate/stop button click."""
        state = self._presenter.state

        if state == IntrinsicCalibrationState.COLLECTING:
            self._presenter.stop_calibration()
        elif state in (IntrinsicCalibrationState.READY, IntrinsicCalibrationState.CALIBRATED):
            # Reset display state when starting new calibration
            self._undistort_checkbox.setChecked(False)
            self._results_display.reset()
            self._presenter.start_calibration()

    def _on_undistort_toggled(self, checked: bool) -> None:
        """Handle undistort checkbox toggle."""
        self._render_thread.set_undistort(checked, self._presenter.calibrated_camera)

        # Show/hide boundary legend based on whether boundary is drawn
        if self._render_thread.shows_boundary:
            self._boundary_legend.show()
        else:
            self._boundary_legend.hide()

        self._render_thread.rerender_cached()

    def _on_calibration_complete(self, output: IntrinsicCalibrationOutput) -> None:
        """Handle successful calibration."""
        # Populate results display FIRST (before state change shows it)
        self._results_display.update_from_output(output)

        # Auto-enable undistort to show the calibration effect
        self._undistort_checkbox.setChecked(True)

    def _on_calibration_failed(self, error_msg: str) -> None:
        """Handle calibration failure."""
        # Could show error in UI, but for now just log it
        logger.error(f"Calibration failed: {error_msg}")

    def closeEvent(self, event) -> None:
        """Clean up on close."""
        self._render_thread.stop()
        self._render_thread.wait(2000)
        super().closeEvent(event)
