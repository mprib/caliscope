"""Presenter for intrinsic camera calibration workflow.

Coordinates the collection of charuco corner observations and calibration
via the domain's pure functions. Emits raw FramePackets for the View to
handle display transforms (undistortion, rotation, padding).

This is a "scratchpad" presenter - accumulated data and calibration results
are transient until emitted to the Controller for persistence.
"""

import logging
from enum import Enum, auto
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from caliscope.cameras.camera_array import CameraData
from caliscope.core.calibrate_intrinsics import (
    IntrinsicCalibrationResult,
    calibrate_intrinsics,
)
from caliscope.core.frame_selector import select_calibration_frames
from caliscope.core.point_data import ImagePoints
from caliscope.packets import FramePacket, PointPacket
from caliscope.recording import FramePacketPublisher, create_publisher
from caliscope.task_manager.cancellation import CancellationToken
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.task_manager.task_manager import TaskManager
from caliscope.task_manager.task_state import TaskState
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


class PresenterState(Enum):
    """Workflow states for intrinsic calibration.

    States are computed from internal reality, not stored separately.
    This prevents state/reality divergence.
    """

    READY = auto()  # Initial state, can start collection
    COLLECTING = auto()  # Video playing, accumulating points
    CALIBRATING = auto()  # calibrate_intrinsics() running via TaskManager
    CALIBRATED = auto()  # Result available, can toggle undistortion


class IntrinsicCalibrationPresenter(QObject):
    """Presenter for single-camera intrinsic calibration workflow.

    Manages the collection of charuco observations from recorded video and
    submission of calibration to TaskManager. Exposes a display_queue for
    the View's processing thread to consume directly (avoids GUI thread hop).

    Signals:
        state_changed: Emitted when computed state changes. View updates UI.
        calibration_complete: Emitted when calibration succeeds. Contains
            a new CameraData with calibration results applied.
        calibration_failed: Emitted when calibration fails. Contains error message.

    Queue:
        display_queue: View's processing thread reads FramePackets from here.
            Keeps heavy frame data off the GUI thread until processed into QPixmap.
    """

    state_changed = Signal(PresenterState)
    calibration_complete = Signal(CameraData)  # Calibrated camera, ready to use
    calibration_failed = Signal(str)

    def __init__(
        self,
        camera: CameraData,
        video_path: Path,
        tracker: Tracker,
        task_manager: TaskManager,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the presenter.

        Args:
            camera: CameraData with port, size, rotation_count
            video_path: Path to the video file for this camera
            tracker: CharucoTracker for point detection
            task_manager: TaskManager for background calibration
            parent: Optional Qt parent
        """
        super().__init__(parent)

        self._camera = camera
        self._video_path = video_path
        self._tracker = tracker
        self._task_manager = task_manager

        # Derived properties for convenience
        self._port = camera.port
        self._image_size = camera.size

        # Scratchpad state
        self._collected_points: list[tuple[int, PointPacket]] = []
        self._calibrated_camera: CameraData | None = None
        self._calibration_task: TaskHandle | None = None

        # Collection state
        self._publisher: FramePacketPublisher | None = None
        self._stream_handle: TaskHandle | None = None
        self._frame_queue: Queue[FramePacket] = Queue()  # From FramePacketPublisher
        self._display_queue: Queue[FramePacket | None] = Queue()  # For View consumption
        self._consumer_thread: Thread | None = None
        self._stop_event = Event()
        self._is_collecting = False

        # Load initial frame so View has something to display
        self._load_initial_frame()

    @property
    def state(self) -> PresenterState:
        """Compute current state from internal reality - never stale."""
        if self._calibrated_camera is not None:
            return PresenterState.CALIBRATED

        if self._calibration_task is not None and self._calibration_task.state == TaskState.RUNNING:
            return PresenterState.CALIBRATING

        if self._is_collecting:
            return PresenterState.COLLECTING

        return PresenterState.READY

    @property
    def display_queue(self) -> Queue[FramePacket | None]:
        """Queue for View's processing thread to consume frames from.

        None sentinel signals end of current sequence (e.g., after stop).
        """
        return self._display_queue

    @property
    def calibrated_camera(self) -> CameraData | None:
        """Access calibrated camera for View's undistortion setup."""
        return self._calibrated_camera

    @property
    def camera(self) -> CameraData:
        """Access original camera data for View's display setup."""
        return self._camera

    def refresh_display(self) -> None:
        """Put a fresh frame on the display queue.

        Call this when display settings change (e.g., undistort toggle)
        and the View needs to re-render with new settings.
        """
        self._load_initial_frame()

    def _load_initial_frame(self) -> None:
        """Read first frame from video and put on display queue."""
        cap = cv2.VideoCapture(str(self._video_path))
        success, frame = cap.read()
        cap.release()

        if success:
            # Note: FramePacket.frame typed as float64 but cv2 returns uint8
            initial_packet = FramePacket(
                port=self._port,
                frame_index=0,
                frame_time=0.0,
                frame=np.asarray(frame),  # type: ignore[arg-type]
                points=None,
            )
            self._display_queue.put(initial_packet)
            logger.debug(f"Loaded initial frame for port {self._port}")
        else:
            logger.warning(f"Failed to load initial frame from {self._video_path}")

    def start_calibration(self) -> None:
        """Start collecting calibration frames.

        Creates FramePacketPublisher with tracker, subscribes to queue,
        and starts playback. Transitions to COLLECTING state.
        """
        if self.state not in (PresenterState.READY, PresenterState.CALIBRATED):
            logger.warning(f"Cannot start calibration in state {self.state}")
            return

        logger.info(f"Starting calibration collection for port {self._port}")

        # Clear previous attempt's data
        self._collected_points.clear()
        self._calibrated_camera = None
        self._calibration_task = None
        self._stream_handle = None

        # Create publisher with tracker
        self._publisher = create_publisher(
            video_directory=self._video_path.parent,
            port=self._camera.port,
            rotation_count=self._camera.rotation_count,
            tracker=self._tracker,
            break_on_last=True,
        )
        self._publisher.subscribe(self._frame_queue)

        # Start consumer thread
        self._stop_event.clear()
        self._consumer_thread = Thread(target=self._consume_frames, daemon=True)
        self._consumer_thread.start()

        # Mark as collecting before starting playback
        self._is_collecting = True
        self._emit_state_changed()

        # Start playback via TaskManager
        self._stream_handle = self._task_manager.submit(
            self._publisher.play_worker,
            name=f"Publisher port {self._port}",
        )

    def stop_calibration(self) -> None:
        """Stop collection and return to READY state.

        Stops the stream, clears accumulated data, and resets state.
        """
        if self.state != PresenterState.COLLECTING:
            logger.warning(f"Cannot stop calibration in state {self.state}")
            return

        logger.info(f"Stopping calibration collection for port {self._port}")

        # Signal consumer to stop
        self._stop_event.set()

        # Stop stream playback via TaskHandle
        if self._stream_handle is not None:
            self._stream_handle.cancel()

        # Wait for consumer thread
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=2.0)
            self._consumer_thread = None

        # Clean up publisher
        if self._publisher is not None:
            self._publisher.unsubscribe(self._frame_queue)
            self._publisher.close()
            self._publisher = None

        # Drain queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break

        # Clear state
        self._collected_points.clear()
        self._is_collecting = False
        self._emit_state_changed()

        # Reload initial frame for display
        self._load_initial_frame()

    def _consume_frames(self) -> None:
        """Pull frames from queue, accumulate points, emit for display.

        Runs in a separate thread. Exits when stop_event is set or
        end-of-stream packet is received.
        """
        logger.debug(f"Consumer thread started for port {self._port}")

        while not self._stop_event.is_set():
            try:
                packet: FramePacket = self._frame_queue.get(timeout=0.1)
            except Empty:
                continue

            # End of stream signal
            if packet.frame_index == -1:
                logger.info(f"End of stream reached for port {self._port}")
                self._on_collection_complete()
                break

            # Accumulate points (store frame_index with PointPacket)
            # Skip empty detections - PointPacket exists but has no points
            if packet.points is not None and len(packet.points.point_id) > 0:
                self._collected_points.append((packet.frame_index, packet.points))

            # Put on display queue for View's processing thread
            self._display_queue.put(packet)

        logger.debug(f"Consumer thread exiting for port {self._port}")

    def _on_collection_complete(self) -> None:
        """Called when video playback finishes. Submits calibration task."""
        self._is_collecting = False

        if len(self._collected_points) == 0:
            logger.warning(f"No points collected for port {self._port}")
            self.calibration_failed.emit("No charuco boards detected in video")
            self._emit_state_changed()
            return

        logger.info(f"Collection complete for port {self._port}: {len(self._collected_points)} frames with points")

        # Build ImagePoints from collected data
        try:
            image_points = self._build_image_points()
        except Exception as e:
            logger.error(f"Failed to build ImagePoints: {e}")
            self.calibration_failed.emit(str(e))
            self._emit_state_changed()
            return

        # Select calibration frames
        selection_result = select_calibration_frames(image_points, self._port, self._image_size)

        if not selection_result.selected_frames:
            logger.warning(f"No frames selected for calibration at port {self._port}")
            self.calibration_failed.emit("Frame selection found no suitable frames")
            self._emit_state_changed()
            return

        logger.info(f"Selected {len(selection_result.selected_frames)} frames for calibration")

        # Submit calibration to TaskManager
        def calibration_worker(token: CancellationToken, handle: TaskHandle) -> IntrinsicCalibrationResult:
            return calibrate_intrinsics(
                image_points,
                self._port,
                self._image_size,
                selection_result.selected_frames,
            )

        self._calibration_task = self._task_manager.submit(
            calibration_worker,
            name=f"Intrinsic calibration port {self._port}",
        )
        self._calibration_task.completed.connect(self._on_calibration_complete)
        self._calibration_task.failed.connect(self._on_calibration_failed)

        self._emit_state_changed()

    def _build_image_points(self) -> ImagePoints:
        """Convert accumulated (frame_index, PointPacket) to ImagePoints."""
        import pandas as pd

        rows = []
        for frame_index, points in self._collected_points:
            # For single-camera calibration, sync_index == frame_index
            point_count = len(points.point_id)

            # Build row data matching ImagePoints schema
            row_data = {
                "sync_index": [frame_index] * point_count,
                "port": [self._port] * point_count,
                "frame_index": [frame_index] * point_count,
                "frame_time": [0.0] * point_count,  # Not used for calibration
                "point_id": points.point_id.tolist(),
                "img_loc_x": points.img_loc[:, 0].tolist(),
                "img_loc_y": points.img_loc[:, 1].tolist(),
                "obj_loc_x": points.obj_loc[:, 0].tolist() if points.obj_loc is not None else [None] * point_count,
                "obj_loc_y": points.obj_loc[:, 1].tolist() if points.obj_loc is not None else [None] * point_count,
                "obj_loc_z": points.obj_loc[:, 2].tolist() if points.obj_loc is not None else [None] * point_count,
            }
            rows.append(pd.DataFrame(row_data))

        df = pd.concat(rows, ignore_index=True)
        return ImagePoints(df)

    def _on_calibration_complete(self, result: IntrinsicCalibrationResult) -> None:
        """Handle successful calibration. Creates calibrated CameraData."""
        logger.info(
            f"Calibration complete for port {self._port}: "
            f"error={result.reprojection_error:.4f}px, "
            f"frames={result.frames_used}"
        )

        # Create new CameraData with calibration results applied
        self._calibrated_camera = CameraData(
            port=self._camera.port,
            size=self._camera.size,
            rotation_count=self._camera.rotation_count,
            error=result.reprojection_error,
            matrix=result.camera_matrix,
            distortions=result.distortions,
            grid_count=result.frames_used,
        )

        self.calibration_complete.emit(self._calibrated_camera)
        self._emit_state_changed()

        # Reload initial frame so View has something to display/undistort
        self._load_initial_frame()

    def _on_calibration_failed(self, exc_type: str, message: str) -> None:
        """Handle calibration failure."""
        logger.error(f"Calibration failed for port {self._port}: {exc_type}: {message}")
        self.calibration_failed.emit(f"{exc_type}: {message}")
        self._emit_state_changed()

    def _emit_state_changed(self) -> None:
        """Emit state_changed signal with current computed state."""
        current_state = self.state
        logger.debug(f"State changed to {current_state} for port {self._port}")
        self.state_changed.emit(current_state)

    def cleanup(self) -> None:
        """Clean up resources. Call before discarding presenter."""
        if self._stream_handle is not None:
            self._stream_handle.cancel()
        if self.state == PresenterState.COLLECTING:
            self.stop_calibration()
