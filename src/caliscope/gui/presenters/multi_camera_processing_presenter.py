"""Presenter for multi-camera synchronized video processing.

Coordinates batch processing of synchronized video to extract 2D landmarks.
Wraps process_synchronized_recording() with Qt signals for UI integration.

This is a "scratchpad" presenter - processing results are transient until
emitted to the Coordinator for persistence.
"""

import logging
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal

from caliscope.cameras.camera_array import CameraData
from caliscope.core.coverage_analysis import (
    ExtrinsicCoverageReport,
    analyze_multi_camera_coverage,
)
from caliscope.core.point_data import ImagePoints
from caliscope.core.process_synchronized_recording import (
    FrameData,
    get_initial_thumbnails,
    process_synchronized_recording,
)
from caliscope.task_manager.cancellation import CancellationToken
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.task_manager.task_manager import TaskManager
from caliscope.task_manager.task_state import TaskState
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


class MultiCameraProcessingState(Enum):
    """Workflow states for multi-camera processing.

    States are computed from internal reality, not stored separately.
    This prevents state/reality divergence.
    """

    UNCONFIGURED = auto()  # Missing required inputs
    READY = auto()  # Can start processing
    PROCESSING = auto()  # Background task running
    COMPLETE = auto()  # Results available


class MultiCameraProcessingPresenter(QObject):
    """Presenter for multi-camera synchronized video processing.

    Manages the extraction of 2D landmarks from synchronized multi-camera
    video recordings. Provides rotation control per camera and throttled
    thumbnail updates during processing.

    Signals:
        state_changed: Emitted when computed state changes. View updates UI.
        progress_updated: Emitted during processing with (current, total, percent).
        thumbnail_updated: Emitted when a camera thumbnail is updated (port, frame).
        processing_complete: Emitted when processing finishes successfully.
            Contains (ImagePoints, ExtrinsicCoverageReport).
        processing_failed: Emitted when processing fails. Contains error message.

    Usage:
        presenter = MultiCameraProcessingPresenter(task_manager, tracker)
        presenter.set_recording_dir(path)
        presenter.set_cameras(cameras)
        presenter.start_processing()  # Emits progress_updated during work
        # On completion: processing_complete emitted with results
    """

    # State signals
    state_changed = Signal(MultiCameraProcessingState)

    # Progress signals
    progress_updated = Signal(int, int, int)  # (current, total, percent)
    thumbnail_updated = Signal(int, object, object)  # (port, NDArray frame, PointPacket | None)

    # Completion signals
    processing_complete = Signal(object, object, object)  # (ImagePoints, ExtrinsicCoverageReport, Tracker)
    processing_failed = Signal(str)  # error message

    # Rotation signals
    rotation_changed = Signal(int, int)  # (port, rotation_count)

    # Thumbnail throttle interval (seconds)
    THUMBNAIL_INTERVAL = 0.1  # ~10 FPS

    def __init__(
        self,
        task_manager: TaskManager,
        tracker: Tracker,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the presenter.

        Args:
            task_manager: TaskManager for background processing
            tracker: Tracker for 2D point extraction
            parent: Optional Qt parent
        """
        super().__init__(parent)

        self._task_manager = task_manager
        self._tracker = tracker

        # Configuration state (set externally)
        self._recording_dir: Path | None = None
        self._cameras: dict[int, CameraData] = {}

        # Processing state (managed internally)
        self._task_handle: TaskHandle | None = None
        self._result: ImagePoints | None = None
        self._coverage_report: ExtrinsicCoverageReport | None = None

        # Thumbnail state
        self._thumbnails: dict[int, NDArray[np.uint8]] = {}
        self._last_thumbnail_time: float = 0.0

    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------

    def _is_task_active(self) -> bool:
        """True if a task is submitted and not yet finished.

        Includes both PENDING (just submitted, worker thread not yet running)
        and RUNNING states. This prevents the race condition where checking
        only RUNNING misses the window between submit() and worker start.
        """
        return self._task_handle is not None and self._task_handle.state in (
            TaskState.PENDING,
            TaskState.RUNNING,
        )

    @property
    def state(self) -> MultiCameraProcessingState:
        """Compute current state from internal reality - never stale."""
        if self._result is not None:
            return MultiCameraProcessingState.COMPLETE

        if self._is_task_active():
            return MultiCameraProcessingState.PROCESSING

        if self._recording_dir is None or not self._cameras:
            return MultiCameraProcessingState.UNCONFIGURED

        return MultiCameraProcessingState.READY

    @property
    def recording_dir(self) -> Path | None:
        """Current recording directory."""
        return self._recording_dir

    @property
    def cameras(self) -> dict[int, CameraData]:
        """Current camera configuration (copy)."""
        return dict(self._cameras)

    @property
    def result(self) -> ImagePoints | None:
        """Processing result (ImagePoints) if complete."""
        return self._result

    @property
    def coverage_report(self) -> ExtrinsicCoverageReport | None:
        """Coverage analysis report if processing complete."""
        return self._coverage_report

    @property
    def thumbnails(self) -> dict[int, NDArray[np.uint8]]:
        """Current thumbnails by port (copy)."""
        return dict(self._thumbnails)

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------

    def set_recording_dir(self, path: Path) -> None:
        """Set the recording directory.

        Resets any existing results and loads initial thumbnails.

        Args:
            path: Directory containing port_N.mp4 and frame_timestamps.csv
        """
        if self.state == MultiCameraProcessingState.PROCESSING:
            logger.warning("Cannot change recording_dir while processing")
            return

        self._recording_dir = path
        self._reset_results()
        self._load_initial_thumbnails()
        self._emit_state_changed()

    def set_cameras(self, cameras: dict[int, CameraData]) -> None:
        """Set the camera configuration.

        Makes a shallow copy of the cameras dict. Resets any existing results
        and loads initial thumbnails.

        Args:
            cameras: Camera data by port
        """
        if self.state == MultiCameraProcessingState.PROCESSING:
            logger.warning("Cannot change cameras while processing")
            return

        # Shallow copy - rotation_count may be modified via set_rotation()
        self._cameras = {port: cam for port, cam in cameras.items()}
        self._reset_results()
        self._load_initial_thumbnails()
        self._emit_state_changed()

    # -------------------------------------------------------------------------
    # Rotation Control
    # -------------------------------------------------------------------------

    def set_rotation(self, port: int, rotation_count: int) -> None:
        """Set the rotation for a camera.

        Updates the local camera copy and emits rotation_changed for coordinator
        to persist. Rotation affects both display orientation and tracker input.

        Args:
            port: Camera port
            rotation_count: Rotation in 90° increments (0=0°, 1=90°, 2=180°, 3=270°)
        """
        # Belt-and-suspenders guard: buttons are disabled, but prevent API misuse
        if self.state == MultiCameraProcessingState.PROCESSING:
            logger.warning("Cannot change rotation while processing")
            return

        if port not in self._cameras:
            logger.warning(f"Cannot set rotation: port {port} not in cameras")
            return

        # Normalize to 0-3 range
        normalized = rotation_count % 4

        # Update local copy (note: shallow copy means this is the same object
        # the coordinator holds; signal notifies it to persist)
        self._cameras[port].rotation_count = normalized

        # Signal for coordinator persistence
        self.rotation_changed.emit(port, normalized)

        # Refresh thumbnail to show new orientation
        self._refresh_thumbnail(port)

        logger.debug(f"Rotation set: port {port} -> {normalized * 90}°")

    # -------------------------------------------------------------------------
    # Processing Control
    # -------------------------------------------------------------------------

    def start_processing(self, subsample: int = 1) -> None:
        """Start background processing.

        Submits process_synchronized_recording to TaskManager.

        Args:
            subsample: Process every Nth sync index (1 = all)
        """
        if self.state != MultiCameraProcessingState.READY:
            logger.warning(f"Cannot start processing in state {self.state}")
            return

        if self._recording_dir is None:
            logger.error("recording_dir is None, cannot start")
            return

        logger.info(f"Starting multi-camera processing: {self._recording_dir}")

        # Clear previous results
        self._reset_results()

        # Capture values for closure
        recording_dir = self._recording_dir
        cameras = dict(self._cameras)
        tracker = self._tracker

        def worker(token: CancellationToken, handle: TaskHandle) -> ImagePoints:
            return process_synchronized_recording(
                recording_dir=recording_dir,
                cameras=cameras,
                tracker=tracker,
                subsample=subsample,
                on_progress=self._on_progress,
                on_frame_data=self._on_frame_data,
                token=token,
            )

        self._task_handle = self._task_manager.submit(
            worker,
            name="Multi-camera processing",
        )
        self._task_handle.completed.connect(self._on_processing_complete)
        self._task_handle.failed.connect(self._on_processing_failed)
        self._task_handle.cancelled.connect(self._on_processing_cancelled)

        self._emit_state_changed()

    def cancel_processing(self) -> None:
        """Cancel the current processing task."""
        if self._is_task_active() and self._task_handle is not None:
            logger.info("Cancelling multi-camera processing")
            self._task_handle.cancel()

    def reset(self) -> None:
        """Reset to READY state, clearing results.

        Does not clear configuration (recording_dir, cameras).
        """
        if self.state == MultiCameraProcessingState.PROCESSING:
            logger.warning("Cannot reset while processing - cancel first")
            return

        self._reset_results()
        self._emit_state_changed()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def update_tracker(self, tracker: Tracker) -> None:
        """Update tracker for next processing run.

        Hot-swaps the tracker reference when charuco config changes. Clears
        any existing results (old tracker's point IDs are stale).

        Cannot be called during PROCESSING state - must cancel first.

        Args:
            tracker: New tracker to use for subsequent processing.
        """
        if self.state == MultiCameraProcessingState.PROCESSING:
            logger.warning("Cannot update tracker while processing")
            return

        self._tracker = tracker
        self._reset_results()
        logger.info("Tracker updated, cleared previous results")
        self._emit_state_changed()

    def cleanup(self) -> None:
        """Clean up resources. Call before discarding presenter."""
        if self._is_task_active() and self._task_handle is not None:
            self._task_handle.cancel()

    # -------------------------------------------------------------------------
    # Private: Callbacks
    # -------------------------------------------------------------------------

    def _on_progress(self, current: int, total: int) -> None:
        """Progress callback from process_synchronized_recording."""
        percent = int(100 * current / total) if total > 0 else 0
        self.progress_updated.emit(current, total, percent)

    def _on_frame_data(self, sync_index: int, frame_data: dict[int, FrameData]) -> None:
        """Frame data callback from process_synchronized_recording.

        Throttled to ~10 FPS to avoid overwhelming the UI.
        """
        now = time.time()
        if now - self._last_thumbnail_time < self.THUMBNAIL_INTERVAL:
            return

        self._last_thumbnail_time = now

        # Update thumbnails for all ports in this sync packet
        # Points passed to View for overlay rendering (MVP: View renders)
        for port, data in frame_data.items():
            self._thumbnails[port] = data.frame
            self.thumbnail_updated.emit(port, data.frame, data.points)

    def _on_processing_complete(self, image_points: ImagePoints) -> None:
        """Handle successful processing completion."""
        logger.info(f"Multi-camera processing complete: {len(image_points.df)} points")

        # Compute coverage analysis
        coverage_report = analyze_multi_camera_coverage(image_points)

        self._result = image_points
        self._coverage_report = coverage_report

        # Emit results with tracker so caller knows where to persist
        self.processing_complete.emit(image_points, coverage_report, self._tracker)
        self._emit_state_changed()

    def _on_processing_failed(self, exc_type: str, message: str) -> None:
        """Handle processing failure."""
        logger.error(f"Multi-camera processing failed: {exc_type}: {message}")
        self.processing_failed.emit(f"{exc_type}: {message}")
        self._emit_state_changed()

    def _on_processing_cancelled(self) -> None:
        """Handle processing cancellation."""
        logger.info("Multi-camera processing cancelled")
        self._emit_state_changed()

    # -------------------------------------------------------------------------
    # Private: Helpers
    # -------------------------------------------------------------------------

    def _reset_results(self) -> None:
        """Clear processing results (not configuration)."""
        self._result = None
        self._coverage_report = None
        self._task_handle = None

    def _emit_state_changed(self) -> None:
        """Emit state_changed signal with current computed state."""
        current_state = self.state
        logger.debug(f"State changed to {current_state}")
        self.state_changed.emit(current_state)

    def _load_initial_thumbnails(self) -> None:
        """Load first frame from each camera for thumbnail display.

        Called when recording_dir or cameras are set. Guards ensure
        both are configured before attempting to load.
        """
        if self._recording_dir is None or not self._cameras:
            return

        try:
            thumbnails = get_initial_thumbnails(self._recording_dir, self._cameras)
            self._thumbnails = thumbnails

            # Emit signal for each loaded thumbnail (no points for initial frames)
            for port, frame in thumbnails.items():
                self.thumbnail_updated.emit(port, frame, None)

            logger.debug(f"Loaded initial thumbnails for {len(thumbnails)} cameras")

        except Exception as e:
            logger.warning(f"Failed to load initial thumbnails: {e}")

    def _refresh_thumbnail(self, port: int) -> None:
        """Refresh thumbnail for a single camera port.

        Used after rotation change to re-emit the cached frame.
        The View applies rotation, so we just need to trigger a re-display.
        No disk I/O - uses cached frame for instant response.
        """
        if port not in self._thumbnails:
            return

        # Re-emit cached frame - View will apply current rotation
        self.thumbnail_updated.emit(port, self._thumbnails[port], None)
        logger.debug(f"Refreshed thumbnail for port {port} (from cache)")
