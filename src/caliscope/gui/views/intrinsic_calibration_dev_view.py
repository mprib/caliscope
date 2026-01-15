"""Minimal development View for intrinsic calibration testing.

This is a development harness to validate the IntrinsicCalibrationPresenter.
Not intended for production use - serves as a testbed for the Presenter's
workflow before building the production View.
"""

import logging
from queue import Empty, Queue
from threading import Event

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
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
from caliscope.packets import FramePacket

logger = logging.getLogger(__name__)


class FrameProcessingThread(QThread):
    """Processes raw frames for display - runs off GUI thread.

    Reads directly from Presenter's display_queue (no intermediate signal).
    Applies display transforms and emits QPixmaps for the GUI thread.
    """

    pixmap_ready = Signal(QPixmap)

    def __init__(
        self,
        display_queue: Queue[FramePacket | None],
        camera: CameraData,
        pixmap_edge_length: int = 500,
        parent: QThread | None = None,
    ):
        super().__init__(parent)
        self._display_queue = display_queue
        self._camera = camera
        self._pixmap_edge_length = pixmap_edge_length
        self._undistort_enabled = False
        self._visualizer: LensModelVisualizer | None = None
        self._keep_running = Event()

    def set_undistort(self, enabled: bool, calibrated_camera: CameraData | None) -> None:
        """Enable/disable undistortion."""
        self._undistort_enabled = enabled

        # Create visualizer on first enable
        if enabled and self._visualizer is None and calibrated_camera is not None:
            self._visualizer = LensModelVisualizer(calibrated_camera)
            logger.info(f"Created LensModelVisualizer for port {calibrated_camera.port}")

    @property
    def shows_boundary(self) -> bool:
        """True if the visualizer draws the original frame boundary."""
        if self._visualizer is None or not self._undistort_enabled:
            return False
        return self._visualizer.content_expands_beyond_frame

    def stop(self) -> None:
        """Signal thread to stop."""
        self._keep_running.clear()

    def run(self) -> None:
        """Main processing loop - reads directly from Presenter's queue."""
        self._keep_running.set()
        logger.debug(f"Frame processing thread started for port {self._camera.port}")

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

            # Processing pipeline (mirrors PlaybackFrameEmitter)
            frame = packet.frame_with_points
            if frame is None:
                continue

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

        logger.debug(f"Frame processing thread exiting for port {self._camera.port}")


class IntrinsicCalibrationDevView(QWidget):
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
        self._setup_processing_thread()
        self._connect_signals()

        # Initial UI state
        self._update_ui_for_state(presenter.state)

    def _setup_ui(self) -> None:
        """Create UI elements."""
        layout = QVBoxLayout(self)

        # Frame display
        self._frame_label = QLabel()
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumSize(500, 500)
        self._frame_label.setStyleSheet("background-color: #1a1a1a;")
        layout.addWidget(self._frame_label)

        # Legend for boundary overlay (hidden by default)
        self._boundary_legend = QLabel("┈┈ Original frame boundary")
        self._boundary_legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._boundary_legend.setStyleSheet("color: #00FFFF;")  # Cyan to match boundary
        self._boundary_legend.hide()
        layout.addWidget(self._boundary_legend)

        # Status label
        self._status_label = QLabel("Status: READY")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        # Position slider row
        slider_row = QHBoxLayout()

        self._position_slider = QSlider(Qt.Orientation.Horizontal)
        self._position_slider.setMinimum(0)
        self._position_slider.setMaximum(max(0, self._presenter.frame_count - 1))
        slider_row.addWidget(self._position_slider)

        self._frame_counter = QLabel(f"0 / {self._presenter.frame_count - 1}")
        self._frame_counter.setMinimumWidth(100)
        slider_row.addWidget(self._frame_counter)

        layout.addLayout(slider_row)

        # Controls row
        controls = QHBoxLayout()

        self._calibrate_btn = QPushButton("Calibrate")
        self._calibrate_btn.clicked.connect(self._on_calibrate_clicked)
        controls.addWidget(self._calibrate_btn)

        self._undistort_checkbox = QCheckBox("Undistort")
        self._undistort_checkbox.setEnabled(False)
        self._undistort_checkbox.toggled.connect(self._on_undistort_toggled)
        controls.addWidget(self._undistort_checkbox)

        layout.addLayout(controls)

    def _setup_processing_thread(self) -> None:
        """Create and start the frame processing thread."""
        self._processing_thread = FrameProcessingThread(
            display_queue=self._presenter.display_queue,
            camera=self._presenter.camera,
        )
        self._processing_thread.pixmap_ready.connect(self._on_pixmap_ready)
        self._processing_thread.start()

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
            # Reset undistort when starting new calibration
            self._undistort_checkbox.setChecked(False)
            self._presenter.start_calibration()

    def _on_undistort_toggled(self, checked: bool) -> None:
        """Handle undistort checkbox toggle."""
        self._processing_thread.set_undistort(checked, self._presenter.calibrated_camera)

        # Show/hide boundary legend based on whether boundary is drawn
        if self._processing_thread.shows_boundary:
            self._boundary_legend.show()
        else:
            self._boundary_legend.hide()

        # Request fresh frame to show undistort effect
        self._presenter.refresh_display()

    def _on_calibration_complete(self, calibrated_camera: CameraData) -> None:
        """Handle successful calibration."""
        error = calibrated_camera.error or 0.0
        grid_count = calibrated_camera.grid_count or 0
        self._status_label.setText(f"Status: CALIBRATED (RMSE: {error:.3f}px, frames: {grid_count})")

        # Auto-enable undistortion to show calibration effect
        self._undistort_checkbox.setChecked(True)

    def _on_calibration_failed(self, error_msg: str) -> None:
        """Handle calibration failure."""
        self._status_label.setText(f"Status: FAILED - {error_msg}")

    def closeEvent(self, event) -> None:
        """Clean up on close."""
        self._processing_thread.stop()
        self._processing_thread.wait(2000)
        super().closeEvent(event)
