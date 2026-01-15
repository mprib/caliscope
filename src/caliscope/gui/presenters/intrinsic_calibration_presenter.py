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
from threading import Event, Lock, Thread

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from caliscope.cameras.camera_array import CameraData
from caliscope.core.calibrate_intrinsics import (
    IntrinsicCalibrationResult,
    calibrate_intrinsics,
)
from caliscope.core.frame_selector import FrameSelectionResult, select_calibration_frames
from caliscope.core.point_data import ImagePoints
from caliscope.packets import FramePacket, PointPacket
from caliscope.recording import create_publisher
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

    A single FramePacketPublisher lives across all states, enabling scrubbing
    in READY/CALIBRATED states and collection during COLLECTING state.

    Signals:
        state_changed: Emitted when computed state changes. View updates UI.
        calibration_complete: Emitted when calibration succeeds. Contains
            a new CameraData with calibration results applied.
        calibration_failed: Emitted when calibration fails. Contains error message.
        frame_position_changed: Emitted when current frame changes (background thread,
            Qt.AutoConnection queues to main thread).

    Queue:
        display_queue: View's processing thread reads FramePackets from here.
            Keeps heavy frame data off the GUI thread until processed into QPixmap.
    """

    state_changed = Signal(PresenterState)
    calibration_complete = Signal(CameraData)  # Calibrated camera, ready to use
    calibration_failed = Signal(str)
    frame_position_changed = Signal(int)  # Current frame index

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
        self._selection_result: FrameSelectionResult | None = None

        # Lock for thread-safe access to collected_points from View
        self._overlay_lock = Lock()

        # Display queue for View consumption
        self._display_queue: Queue[FramePacket | None] = Queue()

        # Collection state
        self._is_collecting = False

        # Single publisher for all states (scrubbing + collection)
        self._publisher = create_publisher(
            video_directory=self._video_path.parent,
            port=self._camera.port,
            rotation_count=self._camera.rotation_count,
            tracker=self._tracker,
            break_on_last=False,  # Pause at end instead of exit
        )
        self._frame_queue: Queue[FramePacket] = Queue()
        self._publisher.subscribe(self._frame_queue)

        # Start publisher worker (will read first frame, then we pause)
        self._stream_handle = self._task_manager.submit(
            self._publisher.play_worker,
            name=f"Publisher port {self._port}",
        )
        self._publisher.pause()  # Immediately pause for scrubbing mode

        # Guaranteed initial frame display (don't rely on thread timing)
        self._load_initial_frame()

        # Consumer thread runs continuously
        self._stop_event = Event()
        self._consumer_thread = Thread(target=self._consume_frames, daemon=True)
        self._consumer_thread.start()

        # Position tracking
        self._current_frame_index: int = 0

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

    @property
    def frame_count(self) -> int:
        """Total frames in video."""
        return self._publisher.last_frame_index + 1

    @property
    def current_frame_index(self) -> int:
        """Current frame position."""
        return self._current_frame_index

    @property
    def collected_points(self) -> list[tuple[int, PointPacket]]:
        """Thread-safe access to accumulated points for overlay rendering."""
        with self._overlay_lock:
            return list(self._collected_points)

    @property
    def selected_frame_indices(self) -> list[int] | None:
        """Frame indices used in calibration, or None if not yet calibrated."""
        if self._selection_result is None:
            return None
        return self._selection_result.selected_frames

    @property
    def board_connectivity(self) -> set[tuple[int, int]]:
        """Point ID pairs that should be connected to form grid."""
        return self._tracker.get_connected_points()

    def refresh_display(self) -> None:
        """Put a fresh frame on the display queue.

        Call this when display settings change (e.g., undistort toggle)
        and the View needs to re-render with new settings.
        """
        self._load_initial_frame()

    def seek_to(self, frame_index: int) -> None:
        """Seek to frame. Works in READY/CALIBRATED states via publisher's jump_to."""
        if self.state not in (PresenterState.READY, PresenterState.CALIBRATED):
            return

        frame_index = max(0, min(frame_index, self.frame_count - 1))
        self._publisher.jump_to(
            frame_index, exact=True
        )  # exact=False would cause Fast seek, skipping between keyframes

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

        Resets to beginning and unpauses the publisher.
        """
        if self.state not in (PresenterState.READY, PresenterState.CALIBRATED):
            logger.warning(f"Cannot start calibration in state {self.state}")
            return

        logger.info(f"Starting calibration collection for port {self._port}")

        # Set collecting FIRST to block concurrent seeks
        self._is_collecting = True
        self._emit_state_changed()

        # Clear previous attempt's data
        with self._overlay_lock:
            self._collected_points.clear()
        self._calibrated_camera = None
        self._calibration_task = None
        self._selection_result = None

        # Reset to beginning and start playback
        self._publisher.jump_to(0, exact=True)
        self._publisher.unpause()

    def stop_calibration(self) -> None:
        """Stop collection and return to READY state.

        Pauses playback and clears accumulated data.
        """
        if self.state != PresenterState.COLLECTING:
            logger.warning(f"Cannot stop calibration in state {self.state}")
            return

        logger.info(f"Stopping calibration collection for port {self._port}")

        self._publisher.pause()
        with self._overlay_lock:
            self._collected_points.clear()
        self._is_collecting = False
        self._emit_state_changed()

    def _consume_frames(self) -> None:
        """Pull frames from queue, accumulate points when collecting, emit for display.

        Runs continuously across all states. Exits when stop_event is set or
        publisher is cancelled externally.
        """
        logger.debug(f"Consumer thread started for port {self._port}")

        while not self._stop_event.is_set():
            # Exit if publisher was cancelled externally
            if self._stream_handle is not None and self._stream_handle.state == TaskState.CANCELLED:
                break

            try:
                packet: FramePacket = self._frame_queue.get(timeout=0.1)
            except Empty:
                continue

            # Skip end-of-stream markers
            if packet.frame_index == -1:
                continue

            # Accumulate points only during collection
            if self._is_collecting and packet.points is not None and len(packet.points.point_id) > 0:
                with self._overlay_lock:
                    self._collected_points.append((packet.frame_index, packet.points))

            # Always emit for display
            self._display_queue.put(packet)
            self._current_frame_index = packet.frame_index
            self.frame_position_changed.emit(packet.frame_index)

            # Detect end of video during collection
            if self._is_collecting and packet.frame_index >= self._publisher.last_frame_index:
                self._on_collection_complete()

        logger.debug(f"Consumer thread exiting for port {self._port}")

    def _on_collection_complete(self) -> None:
        """Called when video playback finishes. Submits calibration task."""
        self._publisher.pause()
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

        # Store selection result for overlay rendering
        self._selection_result = selection_result

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

        # Jump to first frame so View can display with undistortion
        self._publisher.jump_to(0, exact=True)

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
        # Stop consumer thread
        self._stop_event.set()
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=2.0)

        # Cancel publisher worker
        if self._stream_handle is not None:
            self._stream_handle.cancel()

        # Clean up publisher
        self._publisher.unsubscribe(self._frame_queue)
        self._publisher.close()
