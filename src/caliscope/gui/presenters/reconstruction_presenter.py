"""Presenter for reconstruction workflow (post-processing).

Coordinates the reconstruction of 3D trajectories from recorded video:
1. Stage 1: create_xy() - Synchronized 2D landmark detection
2. Stage 2: create_xyz() - Triangulation and export

This is a state machine presenter following the IntrinsicCalibrationPresenter
pattern. States are computed from reality (task state + file existence).
"""

import logging
from enum import Enum, auto
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from caliscope.cameras.camera_array import CameraArray
from caliscope.post_processing.post_processor import PostProcessor
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.task_manager.task_manager import TaskManager
from caliscope.task_manager.task_state import TaskState
from caliscope.trackers.tracker_enum import TrackerEnum

logger = logging.getLogger(__name__)


class ReconstructionState(Enum):
    """Workflow states for reconstruction.

    States are computed from internal reality, not stored separately.
    This prevents state/reality divergence.
    """

    IDLE = auto()  # Can select recording/tracker, ready to start
    RECONSTRUCTING = auto()  # create_xy or create_xyz running
    COMPLETE = auto()  # xyz output file exists
    ERROR = auto()  # Last attempt failed


class ReconstructionPresenter(QObject):
    """Presenter for reconstruction workflow.

    Manages the selection of recordings and trackers, submission of
    reconstruction tasks to TaskManager, and progress reporting.

    State is computed from reality:
    - Task running → RECONSTRUCTING
    - Task failed or error set → ERROR
    - xyz file exists → COMPLETE
    - Otherwise → IDLE

    Signals:
        state_changed: Emitted when computed state changes
        reconstruction_complete: Emitted with xyz output path on success
        reconstruction_failed: Emitted with error message on failure
        progress_updated: Emitted with (percent, message) during processing
    """

    state_changed = Signal(ReconstructionState)
    reconstruction_complete = Signal(Path)  # xyz output path
    reconstruction_failed = Signal(str)  # error message
    progress_updated = Signal(int, str)  # percent (0-100), message

    def __init__(
        self,
        workspace_dir: Path,
        camera_array: CameraArray,
        task_manager: TaskManager,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the presenter.

        Args:
            workspace_dir: Root workspace directory
            camera_array: Calibrated camera array for triangulation
            task_manager: TaskManager for background processing
            parent: Optional Qt parent
        """
        super().__init__(parent)

        self._workspace_dir = workspace_dir
        self._camera_array = camera_array
        self._task_manager = task_manager

        # Selection state
        self._selected_recording: str | None = None
        self._selected_tracker: TrackerEnum | None = None

        # Task tracking
        self._processing_task: TaskHandle | None = None
        self._last_error: str | None = None

    @property
    def state(self) -> ReconstructionState:
        """Compute current state from internal reality - never stale.

        Priority order (task state takes precedence to avoid race conditions):
        1. Task RUNNING → RECONSTRUCTING
        2. Task FAILED or _last_error set → ERROR
        3. Task CANCELLED → IDLE (cancellation returns to idle)
        4. xyz output exists → COMPLETE
        5. Otherwise → IDLE
        """
        if self._processing_task is not None:
            task_state = self._processing_task.state
            if task_state == TaskState.RUNNING:
                return ReconstructionState.RECONSTRUCTING
            if task_state == TaskState.FAILED:
                return ReconstructionState.ERROR
            # CANCELLED or COMPLETED fall through to file check

        if self._last_error is not None:
            return ReconstructionState.ERROR

        # Check if output exists (requires valid selection)
        output_path = self.xyz_output_path
        if output_path is not None and output_path.exists():
            return ReconstructionState.COMPLETE

        return ReconstructionState.IDLE

    @property
    def available_recordings(self) -> list[str]:
        """List of valid recording directory names."""
        recordings_dir = self._workspace_dir / "recordings"
        if not recordings_dir.exists():
            return []

        # A valid recording has config.toml and at least one .mp4
        valid = []
        for item in recordings_dir.iterdir():
            if item.is_dir():
                has_config = (item / "config.toml").exists()
                has_video = any(item.glob("*.mp4"))
                if has_config and has_video:
                    valid.append(item.name)

        return sorted(valid)

    @property
    def available_trackers(self) -> list[TrackerEnum]:
        """List of available trackers (excludes CHARUCO which is for calibration)."""
        return [t for t in TrackerEnum if t.name != "CHARUCO"]

    @property
    def selected_recording(self) -> str | None:
        """Currently selected recording name."""
        return self._selected_recording

    @property
    def selected_tracker(self) -> TrackerEnum | None:
        """Currently selected tracker."""
        return self._selected_tracker

    @property
    def xyz_output_path(self) -> Path | None:
        """Computed path to xyz output file based on current selection.

        Returns None if recording or tracker not selected.
        """
        if not self._selected_recording or not self._selected_tracker:
            return None

        return (
            self._workspace_dir
            / "recordings"
            / self._selected_recording
            / self._selected_tracker.name
            / f"xyz_{self._selected_tracker.name}.csv"
        )

    @property
    def last_error(self) -> str | None:
        """Last error message, if any."""
        return self._last_error

    def select_recording(self, name: str) -> None:
        """Select a recording for processing.

        Clears any previous error state when selection changes.
        """
        if name not in self.available_recordings:
            logger.warning(f"Recording '{name}' not in available recordings")
            return

        self._selected_recording = name
        self._last_error = None  # Clear error on new selection
        self._processing_task = None  # Clear stale task reference
        self._emit_state_changed()
        logger.info(f"Selected recording: {name}")

    def select_tracker(self, tracker: TrackerEnum) -> None:
        """Select a tracker for processing.

        Clears any previous error state when selection changes.
        """
        if tracker not in self.available_trackers:
            logger.warning(f"Tracker '{tracker.name}' not available")
            return

        self._selected_tracker = tracker
        self._last_error = None  # Clear error on new selection
        self._processing_task = None  # Clear stale task reference
        self._emit_state_changed()
        logger.info(f"Selected tracker: {tracker.name}")

    def start_reconstruction(self) -> None:
        """Start the reconstruction process.

        Requires both recording and tracker to be selected.
        """
        if self.state not in (ReconstructionState.IDLE, ReconstructionState.COMPLETE):
            logger.warning(f"Cannot start reconstruction in state {self.state}")
            return

        if not self._selected_recording or not self._selected_tracker:
            logger.warning("Cannot start: recording or tracker not selected")
            self._last_error = "Recording and tracker must be selected"
            self._emit_state_changed()
            return

        logger.info(
            f"Starting reconstruction: recording={self._selected_recording}, tracker={self._selected_tracker.name}"
        )

        # Clear previous state
        self._last_error = None

        # Capture values for closure
        recording_path = self._workspace_dir / "recordings" / self._selected_recording
        tracker_enum = self._selected_tracker
        camera_array = self._camera_array

        def worker(token, handle):
            processor = PostProcessor(camera_array, recording_path, tracker_enum)

            # Stage 1: 2D landmark detection (0-80%)
            if not processor.create_xy(token=token, handle=handle):
                return None  # Cancelled

            # Stage 2: Triangulation (80-100%)
            handle.report_progress(85, "Stage 2: Triangulating 3D points")
            processor.create_xyz()
            handle.report_progress(100, "Complete")

            return processor

        self._processing_task = self._task_manager.submit(worker, name="reconstruction")

        # Connect signals
        self._processing_task.completed.connect(self._on_reconstruction_complete)
        self._processing_task.failed.connect(self._on_reconstruction_failed)
        self._processing_task.cancelled.connect(self._on_reconstruction_cancelled)
        self._processing_task.progress_updated.connect(self._on_progress)

        self._emit_state_changed()

    def cancel_reconstruction(self) -> None:
        """Cancel the running reconstruction task."""
        if self._processing_task is not None and self._processing_task.state == TaskState.RUNNING:
            logger.info("Cancelling reconstruction")
            self._processing_task.cancel()

    def cleanup(self) -> None:
        """Clean up resources. Call before discarding presenter."""
        self.cancel_reconstruction()

    def _on_reconstruction_complete(self, result: object) -> None:
        """Handle successful reconstruction."""
        output_path = self.xyz_output_path
        logger.info(f"Reconstruction complete: {output_path}")

        self._emit_state_changed()

        if output_path is not None:
            self.reconstruction_complete.emit(output_path)

    def _on_reconstruction_failed(self, exc_type: str, message: str) -> None:
        """Handle reconstruction failure."""
        error_msg = f"{exc_type}: {message}"
        logger.error(f"Reconstruction failed: {error_msg}")

        self._last_error = error_msg
        self._emit_state_changed()
        self.reconstruction_failed.emit(error_msg)

    def _on_reconstruction_cancelled(self) -> None:
        """Handle reconstruction cancellation."""
        logger.info("Reconstruction was cancelled")
        self._emit_state_changed()

    def _on_progress(self, percent: int, message: str) -> None:
        """Forward progress updates to our signal."""
        self.progress_updated.emit(percent, message)

    def _emit_state_changed(self) -> None:
        """Emit state_changed signal with current computed state."""
        current_state = self.state
        logger.debug(f"State changed to {current_state}")
        self.state_changed.emit(current_state)
