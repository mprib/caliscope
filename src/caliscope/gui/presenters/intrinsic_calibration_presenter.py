"""Presenter for intrinsic camera calibration workflow.

Coordinates the collection of calibration board corner observations and calibration
via the domain's pure functions. Emits raw TrackedFrames for the View to
handle display transforms (undistortion, rotation, padding).

This is a "scratchpad" presenter - accumulated data and calibration results
are transient until emitted to the Coordinator for persistence.
"""

import logging
from enum import Enum, auto
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

from PySide6.QtCore import QObject, Qt, Signal

from caliscope.cameras.camera_array import CameraData
from caliscope.core.calibrate_intrinsics import (
    IntrinsicCalibrationOutput,
    IntrinsicCalibrationReport,
    run_intrinsic_calibration,
)
from caliscope.core.frame_selector import IntrinsicCoverageReport, select_calibration_frames
from caliscope.core.point_data import ImagePoints
from caliscope.packets import PointPacket, TrackedFrame
from caliscope.recording.frame_packet_streamer import create_streamer
from caliscope.recording.frame_source import FrameSource
from caliscope.task_manager.cancellation import CancellationToken
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.task_manager.task_manager import TaskManager
from caliscope.task_manager.task_state import TaskState
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


class IntrinsicCalibrationState(Enum):
    """Workflow states for intrinsic calibration.

    States are computed from internal reality, not stored separately.
    This prevents state/reality divergence.
    """

    READY = auto()  # Initial state, can start collection
    COLLECTING = auto()  # Batch loop running, accumulating points
    CALIBRATING = auto()  # calibrate_intrinsics() running via TaskManager
    CALIBRATED = auto()  # Result available, can toggle undistortion


class IntrinsicCalibrationPresenter(QObject):
    """Presenter for single-camera intrinsic calibration workflow.

    Manages the collection of calibration board observations from recorded video and
    submission of calibration to TaskManager. Exposes a display_queue for
    the View's processing thread to consume directly (avoids GUI thread hop).

    Collection decodes forward through the video via FrameSource.next_frame() with
    wanted_indices filtering, skipping BGR conversion on unwanted frames.

    A FramePacketStreamer provides the initial frame for display.

    Signals:
        state_changed: Emitted when computed state changes. View updates UI.
        calibration_complete: Emitted when calibration succeeds. Contains
            a new CameraData with calibration results applied.
        calibration_failed: Emitted when calibration fails. Contains error message.
        frame_position_changed: Emitted when current frame changes (background thread,
            Qt.AutoConnection queues to main thread).

    Queue:
        display_queue: View's processing thread reads TrackedFrames from here.
            Keeps heavy frame data off the GUI thread until processed into QPixmap.
    """

    state_changed = Signal(IntrinsicCalibrationState)
    calibration_complete = Signal(object)  # IntrinsicCalibrationOutput
    calibration_failed = Signal(str)
    frame_position_changed = Signal(int)  # Current frame index

    def __init__(
        self,
        camera: CameraData,
        video_path: Path,
        tracker: Tracker,
        task_manager: TaskManager,
        parent: QObject | None = None,
        restored_report: IntrinsicCalibrationReport | None = None,
        restored_points: list[tuple[int, PointPacket]] | None = None,
        frame_skip: int = 1,
    ) -> None:
        """Initialize the presenter.

        Args:
            camera: CameraData with cam_id, size, rotation_count
            video_path: Path to the video file for this camera
            tracker: Tracker for calibration board point detection
            task_manager: TaskManager for background calibration
            parent: Optional Qt parent
            restored_report: Optional report from previous calibration for overlay restoration
            restored_points: Optional collected points from previous calibration (session-only)
            frame_skip: Process every Nth frame during collection
        """
        super().__init__(parent)

        self._camera = camera
        self._video_path = video_path
        self._tracker = tracker
        self._task_manager = task_manager
        self._frame_skip = frame_skip

        # Derived properties for convenience
        self._cam_id = camera.cam_id
        self._image_size = camera.size

        # Scratchpad state - may be restored from previous calibration
        self._collected_points: list[tuple[int, PointPacket]] = []
        self._output: IntrinsicCalibrationOutput | None = None
        self._calibration_task: TaskHandle | None = None
        self._selection_result: IntrinsicCoverageReport | None = None

        # Restore previous calibration state if available
        if restored_report is not None and camera.matrix is not None:
            self._output = IntrinsicCalibrationOutput(camera=camera, report=restored_report)
            logger.info(f"Restored calibration for cam_id {self._cam_id}")

        if restored_points is not None:
            self._collected_points = list(restored_points)

        # Display queue for View consumption
        self._display_queue: Queue[TrackedFrame | None] = Queue()

        # Collection state
        self._is_collecting = False
        self._stop_collection = Event()
        self._collection_thread: Thread | None = None

        # Streamer reads the first frame for initial display
        self._streamer = create_streamer(
            video_directory=self._video_path.parent,
            cam_id=self._camera.cam_id,
            rotation_count=self._camera.rotation_count,
            tracker=self._tracker,
            end_behavior="pause",  # Pause at end for interactive scrubbing
            pixel_format=self._tracker.pixel_format,
        )
        self._frame_queue: Queue[TrackedFrame] = Queue()
        self._streamer.subscribe(self._frame_queue)

        # Start streamer worker (will read first frame, then we pause)
        self._stream_handle = self._task_manager.submit(
            self._streamer.play_worker,
            name=f"Streamer cam_id {self._cam_id}",
            auto_start=False,
        )
        self._task_manager.start_task(self._stream_handle.task_id)
        self._streamer.pause()

        self._current_frame_index: int = self._streamer.start_frame_index
        self._first_tracked_frame: TrackedFrame | None = None

        # Consumer thread caches first frame and forwards to display
        self._stop_event = Event()
        self._consumer_thread = Thread(target=self._consume_frames, daemon=True)
        self._consumer_thread.start()

    @property
    def state(self) -> IntrinsicCalibrationState:
        """Compute current state from internal reality - never stale."""
        if self._output is not None:
            return IntrinsicCalibrationState.CALIBRATED

        if self._calibration_task is not None and self._calibration_task.state == TaskState.RUNNING:
            return IntrinsicCalibrationState.CALIBRATING

        if self._is_collecting:
            return IntrinsicCalibrationState.COLLECTING

        return IntrinsicCalibrationState.READY

    @property
    def display_queue(self) -> Queue[TrackedFrame | None]:
        """Queue for View's processing thread to consume frames from.

        None sentinel signals end of current sequence (e.g., after stop).
        """
        return self._display_queue

    @property
    def calibrated_camera(self) -> CameraData | None:
        """Access calibrated camera for View's undistortion setup."""
        return self._output.camera if self._output is not None else None

    @property
    def calibration_report(self) -> IntrinsicCalibrationReport | None:
        """Access calibration quality report for display."""
        return self._output.report if self._output is not None else None

    @property
    def camera(self) -> CameraData:
        """Access original camera data for View's display setup."""
        return self._camera

    @property
    def frame_count(self) -> int:
        """Total frames in video."""
        return self._streamer.last_frame_index + 1

    @property
    def current_frame_index(self) -> int:
        """Current frame position."""
        return self._current_frame_index

    @property
    def collected_points(self) -> list[tuple[int, PointPacket]]:
        """Accumulated points for overlay rendering. Returns a copy."""
        return list(self._collected_points)

    @property
    def selected_frame_indices(self) -> list[int] | None:
        """Selected frame indices for overlay rendering. Returns a copy.

        Checks both the selection result (from current calibration) and the
        output report (from restored calibration) for the frame list.
        """
        if self._selection_result is not None:
            return list(self._selection_result.selected_frames)
        if self._output is not None:
            return list(self._output.report.selected_frames)
        return None

    @property
    def board_connectivity(self) -> set[tuple[int, int]]:
        """Point ID pairs that should be connected to form grid."""
        return self._tracker.get_connected_points()

    def refresh_display(self) -> None:
        """Put a fresh frame on the display queue.

        Call this when display settings change (e.g., undistort toggle)
        and the View needs to re-render with new settings.
        """
        self._show_first_frame()

    def _show_first_frame(self) -> None:
        """Put the cached first frame on the display queue."""
        if self._first_tracked_frame is not None:
            self._display_queue.put(self._first_tracked_frame)

    def start_calibration(self) -> None:
        """Start collecting calibration frames via forward decode."""
        if self.state not in (IntrinsicCalibrationState.READY, IntrinsicCalibrationState.CALIBRATED):
            logger.warning(f"Cannot start calibration in state {self.state}")
            return

        logger.info(f"Starting calibration collection for cam_id {self._cam_id}")

        # Clear previous calibration data BEFORE setting collecting flag
        # (state is computed: CALIBRATED check comes before COLLECTING check)
        self._collected_points.clear()
        self._selection_result = None
        self._output = None
        self._calibration_task = None

        # Set collecting and emit state change
        self._is_collecting = True
        self._stop_collection.clear()
        self._emit_state_changed()

        # Spawn batch collection thread
        self._collection_thread = Thread(target=self._run_collection, daemon=True)
        self._collection_thread.start()

    def stop_calibration(self) -> None:
        """Stop collection and return to READY state.

        Signals the collection thread to stop and clears accumulated data.
        """
        if self.state != IntrinsicCalibrationState.COLLECTING:
            logger.warning(f"Cannot stop calibration in state {self.state}")
            return

        logger.info(f"Stopping calibration collection for cam_id {self._cam_id}")

        self._stop_collection.set()
        if self._collection_thread is not None:
            self._collection_thread.join(timeout=5.0)
            self._collection_thread = None

        self._collected_points.clear()
        self._is_collecting = False
        self._emit_state_changed()
        self._show_first_frame()

    def _run_collection(self) -> None:
        """Batch collection loop — decodes the video once forward, sampling every Nth frame.

        Creates a temporary FrameSource, does a single forward pass over the
        subsampled frame indices, tracks each, accumulates detected points, and
        emits for display.
        """
        frame_skip = max(1, self._frame_skip)
        wanted = set(range(0, self._streamer.last_frame_index + 1, frame_skip))
        frame_source = FrameSource(
            self._video_path.parent,
            self._cam_id,
            wanted_indices=wanted,
            pixel_format=self._tracker.pixel_format,
        )

        logger.info(
            f"Collection batch: {len(wanted)} frames (skip={frame_skip}, total available={frame_source.frame_count})"
        )

        try:
            while (raw := frame_source.next_frame()) is not None:
                if self._stop_collection.is_set():
                    logger.info(f"Collection cancelled at frame {raw.frame_index}")
                    break

                # Track the frame
                points = self._tracker.get_points(raw.frame, self._cam_id, self._camera.rotation_count)

                # Accumulate if board detected
                if points is not None and len(points.keypoint_id) > 0:
                    self._collected_points.append((raw.frame_index, points))

                # Emit for display
                tracked_frame = TrackedFrame(
                    cam_id=self._cam_id,
                    frame_index=raw.frame_index,
                    frame_time=raw.frame_time,
                    frame=raw.frame,
                    points=points,
                    pixel_format=raw.pixel_format,
                )
                self._display_queue.put(tracked_frame)
                self._current_frame_index = raw.frame_index
                self.frame_position_changed.emit(raw.frame_index)

        finally:
            frame_source.close()

        # If not cancelled, proceed to calibration
        if not self._stop_collection.is_set():
            self._on_collection_complete()

    def _consume_frames(self) -> None:
        """Pull frames from streamer queue, cache the first, and emit for display."""
        logger.debug(f"Consumer thread started for cam_id {self._cam_id}")

        while not self._stop_event.is_set():
            # Exit if streamer was cancelled externally
            if self._stream_handle is not None and self._stream_handle.state == TaskState.CANCELLED:
                break

            try:
                tracked_frame: TrackedFrame = self._frame_queue.get(timeout=0.1)
            except Empty:
                continue

            if tracked_frame.frame_index == -1:
                continue

            if self._first_tracked_frame is None:
                self._first_tracked_frame = tracked_frame

            self._display_queue.put(tracked_frame)
            self._current_frame_index = tracked_frame.frame_index
            self.frame_position_changed.emit(tracked_frame.frame_index)

        logger.debug(f"Consumer thread exiting for cam_id {self._cam_id}")

    # -------------------------------------------------------------------------
    # Post-collection: calibration pipeline
    # -------------------------------------------------------------------------

    def _on_collection_complete(self) -> None:
        """Called when batch loop finishes. Submits calibration task."""
        self._is_collecting = False

        if len(self._collected_points) == 0:
            logger.warning(f"No points collected for cam_id {self._cam_id}")
            self.calibration_failed.emit("No calibration boards detected in video")
            self._emit_state_changed()
            return

        logger.info(f"Collection complete for cam_id {self._cam_id}: {len(self._collected_points)} frames with points")

        # Build ImagePoints from collected data
        try:
            image_points = self._build_image_points()
        except Exception as e:
            logger.error(f"Failed to build ImagePoints: {e}")
            self.calibration_failed.emit(str(e))
            self._emit_state_changed()
            return

        # Select calibration frames
        selection_result = select_calibration_frames(image_points, self._cam_id, self._image_size)

        if not selection_result.selected_frames:
            logger.warning(f"No frames selected for calibration at cam_id {self._cam_id}")
            self.calibration_failed.emit("Frame selection found no suitable frames")
            self._emit_state_changed()
            return

        logger.info(f"Selected {len(selection_result.selected_frames)} frames for calibration")

        # Store selection result for overlay rendering
        self._selection_result = selection_result

        # Capture camera for closure (avoid stale reference)
        camera = self._camera

        # Submit calibration to TaskManager using the orchestrator
        def calibration_worker(token: CancellationToken, handle: TaskHandle) -> IntrinsicCalibrationOutput:
            return run_intrinsic_calibration(
                camera,
                image_points,
                selection_result,
            )

        self._calibration_task = self._task_manager.submit(
            calibration_worker,
            name=f"Intrinsic calibration cam_id {self._cam_id}",
            auto_start=False,
        )
        # Use QueuedConnection - TaskHandle signals emitted from worker threads
        self._calibration_task.completed.connect(
            self._on_calibration_complete,
            Qt.ConnectionType.QueuedConnection,
        )
        self._calibration_task.failed.connect(
            self._on_calibration_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_manager.start_task(self._calibration_task.task_id)

        self._emit_state_changed()

    def _build_image_points(self) -> ImagePoints:
        """Convert accumulated (frame_index, PointPacket) to ImagePoints."""
        import pandas as pd

        rows = []
        for frame_index, points in self._collected_points:
            # For single-camera calibration, sync_index == frame_index
            point_count = len(points.keypoint_id)

            # Build row data matching ImagePoints schema
            row_data = {
                "sync_index": [frame_index] * point_count,
                "cam_id": [self._cam_id] * point_count,
                "frame_index": [frame_index] * point_count,
                "frame_time": [0.0] * point_count,  # Not used for calibration
                "object_id": points.object_id.tolist(),
                "keypoint_id": points.keypoint_id.tolist(),
                "img_loc_x": points.img_loc[:, 0].tolist(),
                "img_loc_y": points.img_loc[:, 1].tolist(),
                "obj_loc_x": points.obj_loc[:, 0].tolist() if points.obj_loc is not None else [None] * point_count,
                "obj_loc_y": points.obj_loc[:, 1].tolist() if points.obj_loc is not None else [None] * point_count,
                "obj_loc_z": points.obj_loc[:, 2].tolist() if points.obj_loc is not None else [None] * point_count,
            }
            rows.append(pd.DataFrame(row_data))

        df = pd.concat(rows, ignore_index=True)
        return ImagePoints(df)

    def _on_calibration_complete(self, output: IntrinsicCalibrationOutput) -> None:
        """Handle successful calibration. Stores output and emits signal."""
        report = output.report
        logger.info(
            f"Calibration complete for cam_id {self._cam_id}: rmse={report.rmse:.3f}px, frames={report.frames_used}"
        )

        # Store complete output (camera + report)
        self._output = output

        self.calibration_complete.emit(output)
        self._emit_state_changed()

        self._show_first_frame()

    def _on_calibration_failed(self, exc_type: str, message: str) -> None:
        """Handle calibration failure."""
        logger.error(f"Calibration failed for cam_id {self._cam_id}: {exc_type}: {message}")
        self.calibration_failed.emit(f"{exc_type}: {message}")
        self._emit_state_changed()

    def _emit_state_changed(self) -> None:
        """Emit state_changed signal with current computed state."""
        current_state = self.state
        logger.debug(f"State changed to {current_state} for cam_id {self._cam_id}")
        self.state_changed.emit(current_state)

    def update_tracker(self, tracker: Tracker) -> None:
        """Update tracker for next calibration run.

        Clears any collected points (old tracker's point IDs are stale) and
        resets calibration state.

        Args:
            tracker: New tracker to use for subsequent calibrations.
        """
        self._tracker = tracker
        self._streamer.update_tracker(tracker)
        self._collected_points.clear()
        self._selection_result = None
        self._output = None
        logger.info(f"Tracker updated for cam_id {self._cam_id}, cleared collected points")
        self._emit_state_changed()
        self.refresh_display()

    @property
    def frame_skip(self) -> int:
        """Current frame skip interval for collection."""
        return self._frame_skip

    def set_frame_skip(self, skip: int) -> None:
        """Set frame skip interval for collection.

        Takes effect on the next collection run. Changing during collection
        has no effect (batch loop captures frame_skip at start).
        """
        self._frame_skip = max(1, skip)

    def cleanup(self) -> None:
        """Clean up resources. Call before discarding presenter."""
        # Stop collection if running
        self._stop_collection.set()
        if self._collection_thread is not None:
            self._collection_thread.join(timeout=2.0)

        # Stop consumer thread
        self._stop_event.set()
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=2.0)

        # Cancel streamer worker
        if self._stream_handle is not None:
            self._stream_handle.cancel()

        # Clean up streamer
        self._streamer.unsubscribe(self._frame_queue)
        self._streamer.close()
