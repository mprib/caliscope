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
from caliscope.gui.frame_emitters.tools import (
    apply_rotation,
    cv2_to_qlabel,
    resize_to_square,
)
from caliscope.gui.lens_model_visualizer import LensModelVisualizer
from caliscope.gui.presenters.intrinsic_calibration_presenter import (
    IntrinsicCalibrationPresenter,
    PresenterState,
)
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
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create the form layout with labeled value fields."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Fit quality group
        fit_group = QGroupBox("Fit Quality")
        fit_layout = QFormLayout(fit_group)
        self._rmse_label = QLabel("—")
        self._grid_count_label = QLabel("—")
        fit_layout.addRow("RMSE:", self._rmse_label)
        fit_layout.addRow("Frames used:", self._grid_count_label)
        layout.addWidget(fit_group)

        # Camera matrix group
        matrix_group = QGroupBox("Camera Matrix")
        matrix_layout = QFormLayout(matrix_group)
        self._fx_label = QLabel("—")
        self._fy_label = QLabel("—")
        self._cx_label = QLabel("—")
        self._cy_label = QLabel("—")
        matrix_layout.addRow("fx:", self._fx_label)
        matrix_layout.addRow("fy:", self._fy_label)
        matrix_layout.addRow("cx:", self._cx_label)
        matrix_layout.addRow("cy:", self._cy_label)
        layout.addWidget(matrix_group)

        # Distortion coefficients group
        dist_group = QGroupBox("Distortion")
        dist_layout = QFormLayout(dist_group)
        self._k1_label = QLabel("—")
        self._k2_label = QLabel("—")
        self._p1_label = QLabel("—")
        self._p2_label = QLabel("—")
        self._k3_label = QLabel("—")
        dist_layout.addRow("k1:", self._k1_label)
        dist_layout.addRow("k2:", self._k2_label)
        dist_layout.addRow("p1:", self._p1_label)
        dist_layout.addRow("p2:", self._p2_label)
        dist_layout.addRow("k3:", self._k3_label)
        layout.addWidget(dist_group)

    def update_from_camera(self, camera: CameraData) -> None:
        """Populate display from calibrated CameraData.

        Args:
            camera: CameraData with calibration results (matrix, distortions, error).
        """
        # Fit quality
        error = camera.error if camera.error is not None else 0.0
        grid_count = camera.grid_count if camera.grid_count is not None else 0
        self._rmse_label.setText(f"{error:.3f} px")
        self._grid_count_label.setText(str(grid_count))

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
        self._rmse_label.setText("—")
        self._grid_count_label.setText("—")
        self._fx_label.setText("—")
        self._fy_label.setText("—")
        self._cx_label.setText("—")
        self._cy_label.setText("—")
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
            logger.info(f"Created LensModelVisualizer for port {calibrated_camera.port}")

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
        else:
            logger.debug(f"rerender_cached called but no cached packet for port {self._camera.port}")

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
    - Status label showing current state
    - Calibrate/Stop button
    - Undistort checkbox (enabled after calibration)
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

    def _setup_ui(self) -> None:
        """Create UI elements."""
        # Main horizontal layout: results on left, video+controls on right
        main_layout = QHBoxLayout(self)

        # Left column: Calibration results (always visible)
        self._results_display = CalibrationResultsDisplay()
        main_layout.addWidget(self._results_display)

        # Right column: Video display and controls
        right_column = QVBoxLayout()

        # Frame display
        self._frame_label = QLabel()
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumSize(500, 500)
        self._frame_label.setStyleSheet("background-color: #1a1a1a;")
        right_column.addWidget(self._frame_label)

        # Legend for boundary overlay (hidden by default)
        self._boundary_legend = QLabel("┈┈ Original frame boundary")
        self._boundary_legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._boundary_legend.setStyleSheet("color: #00FFFF;")  # Cyan to match boundary
        self._boundary_legend.hide()
        right_column.addWidget(self._boundary_legend)

        # Status label
        self._status_label = QLabel("Status: READY")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_column.addWidget(self._status_label)

        # Position slider row
        slider_row = QHBoxLayout()

        self._position_slider = QSlider(Qt.Orientation.Horizontal)
        self._position_slider.setMinimum(0)
        self._position_slider.setMaximum(max(0, self._presenter.frame_count - 1))
        slider_row.addWidget(self._position_slider)

        self._frame_counter = QLabel(f"0 / {self._presenter.frame_count - 1}")
        self._frame_counter.setMinimumWidth(100)
        slider_row.addWidget(self._frame_counter)

        right_column.addLayout(slider_row)

        # Controls row
        controls = QHBoxLayout()

        self._calibrate_btn = QPushButton("Calibrate")
        self._calibrate_btn.clicked.connect(self._on_calibrate_clicked)
        controls.addWidget(self._calibrate_btn)

        self._undistort_checkbox = QCheckBox("Undistort")
        self._undistort_checkbox.setEnabled(False)
        self._undistort_checkbox.toggled.connect(self._on_undistort_toggled)
        controls.addWidget(self._undistort_checkbox)

        right_column.addLayout(controls)

        # Overlay controls row
        overlay_row = QHBoxLayout()

        self._current_points_cb = QCheckBox("Current Points")
        self._current_points_cb.setChecked(True)
        self._current_points_cb.toggled.connect(self._on_overlay_toggled)
        overlay_row.addWidget(self._current_points_cb)

        self._accumulated_cb = QCheckBox("All Points")
        self._accumulated_cb.setChecked(True)
        self._accumulated_cb.toggled.connect(self._on_overlay_toggled)
        overlay_row.addWidget(self._accumulated_cb)

        self._grids_cb = QCheckBox("Selected Grids")
        self._grids_cb.setChecked(True)
        self._grids_cb.setEnabled(False)  # Enable after calibration
        self._grids_cb.toggled.connect(self._on_overlay_toggled)
        overlay_row.addWidget(self._grids_cb)

        right_column.addLayout(overlay_row)

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

    def _update_ui_for_state(self, state: PresenterState) -> None:
        """Update UI elements based on presenter state."""
        self._status_label.setText(f"Status: {state.name}")

        if state == PresenterState.READY:
            self._calibrate_btn.setText("Calibrate")
            self._calibrate_btn.setEnabled(True)
            self._undistort_checkbox.setEnabled(False)
            self._position_slider.setEnabled(True)
        elif state == PresenterState.COLLECTING:
            self._calibrate_btn.setText("Stop")
            self._calibrate_btn.setEnabled(True)
            self._undistort_checkbox.setEnabled(False)
            self._position_slider.setEnabled(False)
        elif state == PresenterState.CALIBRATING:
            self._calibrate_btn.setText("Calibrating...")
            self._calibrate_btn.setEnabled(False)
            self._undistort_checkbox.setEnabled(False)
            self._position_slider.setEnabled(False)
        elif state == PresenterState.CALIBRATED:
            self._calibrate_btn.setText("Recalibrate")
            self._calibrate_btn.setEnabled(True)
            self._undistort_checkbox.setEnabled(True)
            self._position_slider.setEnabled(True)

    def _on_calibrate_clicked(self) -> None:
        """Handle calibrate/stop button click."""
        state = self._presenter.state

        if state == PresenterState.COLLECTING:
            self._presenter.stop_calibration()
        elif state in (PresenterState.READY, PresenterState.CALIBRATED):
            # Reset display state when starting new calibration
            self._undistort_checkbox.setChecked(False)
            self._grids_cb.setChecked(True)
            self._grids_cb.setEnabled(False)
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

        # Re-render cached frame with new settings (don't jump to frame 0)
        self._render_thread.rerender_cached()

    def _on_overlay_toggled(self) -> None:
        """Handle overlay checkbox toggles."""
        self._render_thread.set_overlay_visibility(
            current_points=self._current_points_cb.isChecked(),
            accumulated=self._accumulated_cb.isChecked(),
            selected_grids=self._grids_cb.isChecked(),
        )
        # Re-render cached frame with new settings (don't jump to frame 0)
        self._render_thread.rerender_cached()

    def _on_calibration_complete(self, calibrated_camera: CameraData) -> None:
        """Handle successful calibration."""
        # Populate results display FIRST (before state change shows it)
        self._results_display.update_from_camera(calibrated_camera)

        error = calibrated_camera.error or 0.0
        grid_count = calibrated_camera.grid_count or 0
        self._status_label.setText(f"Status: CALIBRATED (RMSE: {error:.3f}px, frames: {grid_count})")

        # Enable selected grids overlay now that selection is available
        self._grids_cb.setEnabled(True)

        # Auto-enable undistort to show the calibration effect
        self._undistort_checkbox.setChecked(True)

    def _on_calibration_failed(self, error_msg: str) -> None:
        """Handle calibration failure."""
        self._status_label.setText(f"Status: FAILED - {error_msg}")

    def closeEvent(self, event) -> None:
        """Clean up on close."""
        self._render_thread.stop()
        self._render_thread.wait(2000)
        super().closeEvent(event)
