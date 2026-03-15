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

from PySide6.QtCore import QObject, Qt, Signal

from caliscope.cameras.camera_array import CameraArray
from caliscope.gui.geometry.wireframe import WireframeSegment, wireframe_segments_from_view
from caliscope.reconstruction.reconstructor import Reconstructor
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.task_manager.task_manager import TaskManager
from caliscope.task_manager.task_state import TaskState
from caliscope.trackers import tracker_registry

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
    model_download_needed = Signal(object)  # ModelCard when weights missing

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
        self._selected_tracker: str | None = None

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
        """List of valid recording directory names.

        A recording directory is valid if:
        1. It contains a cam_N.mp4 file for every camera in the camera array
        2. There are no unexpected .mp4 files (catches misnamed files)
        """
        recordings_dir = self._workspace_dir / "recordings"
        if not recordings_dir.exists():
            return []

        expected_cam_ids = set(self._camera_array.cameras.keys())
        if not expected_cam_ids:
            return []

        valid = []
        for item in recordings_dir.iterdir():
            if item.is_dir():
                # Get all mp4 files in the directory
                all_mp4s = list(item.glob("*.mp4"))

                # Parse camera IDs from properly named files
                available_cam_ids: set[int] = set()
                for mp4 in all_mp4s:
                    parts = mp4.stem.split("_")
                    if len(parts) == 2 and parts[0] == "cam" and parts[1].isdigit():
                        available_cam_ids.add(int(parts[1]))

                # Valid if: all expected cameras present AND no unexpected mp4 files
                expected_files_count = len(expected_cam_ids)
                if expected_cam_ids == available_cam_ids and len(all_mp4s) == expected_files_count:
                    valid.append(item.name)

        return sorted(valid)

    @property
    def available_trackers(self) -> list[str]:
        """List of available trackers."""
        return tracker_registry.available_names()

    @property
    def selected_recording(self) -> str | None:
        """Currently selected recording name."""
        return self._selected_recording

    @property
    def selected_tracker(self) -> str | None:
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
            / self._selected_tracker
            / f"xyz_{self._selected_tracker}.csv"
        )

    @property
    def last_error(self) -> str | None:
        """Last error message, if any."""
        return self._last_error

    @property
    def historical_camera_array_path(self) -> Path | None:
        """Path to camera_array.toml in current selection's output folder."""
        output_path = self.xyz_output_path
        if output_path is None:
            return None
        return output_path.parent / "camera_array.toml"

    @property
    def camera_array_for_visualization(self) -> CameraArray:
        """Camera array to use for visualization.

        When viewing processed output, returns the historical array from that
        output's folder. Otherwise returns the current calibration.

        This ensures the visualization shows cameras matching the 3D points.
        """
        historical_path = self.historical_camera_array_path
        if historical_path is not None and historical_path.exists():
            try:
                return CameraArray.from_toml(historical_path)
            except Exception:
                logger.warning(f"Failed to load historical camera array from {historical_path}")
        return self._camera_array

    @property
    def is_showing_historical_calibration(self) -> bool:
        """True if visualization is using historical (per-recording) camera array."""
        historical_path = self.historical_camera_array_path
        return historical_path is not None and historical_path.exists() and self.state == ReconstructionState.COMPLETE

    @property
    def camera_array(self) -> CameraArray:
        """Camera array for visualization (delegates to camera_array_for_visualization)."""
        return self.camera_array_for_visualization

    @property
    def wireframe_segments(self) -> list[WireframeSegment] | None:
        """Wireframe segments for the selected tracker, or None."""
        if self._selected_tracker is None:
            return None
        view = tracker_registry.wireframe_for(self._selected_tracker)
        if view is None:
            return None
        return wireframe_segments_from_view(view)

    @property
    def is_tracker_ready(self) -> bool:
        """Check if the selected tracker's model weights are available."""
        if self._selected_tracker is None:
            return False
        return tracker_registry.is_model_ready(self._selected_tracker)

    @property
    def task_manager(self) -> TaskManager:
        """TaskManager instance for background operations."""
        return self._task_manager

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

    def select_tracker(self, tracker: str) -> None:
        """Select a tracker for processing.

        Clears any previous error state when selection changes.
        """
        if tracker not in self.available_trackers:
            logger.warning(f"Tracker '{tracker}' not available")
            return

        self._selected_tracker = tracker
        self._last_error = None  # Clear error on new selection
        self._processing_task = None  # Clear stale task reference
        self._emit_state_changed()
        logger.info(f"Selected tracker: {tracker}")

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

        # Check model readiness (ONNX trackers may have card but no weights)
        if not self.is_tracker_ready:
            card = tracker_registry.model_card_for(self._selected_tracker)
            if card is not None:
                self.model_download_needed.emit(card)
            return

        logger.info(f"Starting reconstruction: recording={self._selected_recording}, tracker={self._selected_tracker}")

        # Clear previous state
        self._last_error = None

        # Capture values for closure
        recording_path = self._workspace_dir / "recordings" / self._selected_recording
        tracker_name = self._selected_tracker
        camera_array = self._camera_array

        def worker(token, handle):
            reconstructor = Reconstructor(camera_array, recording_path, tracker_name)

            # Stage 1: 2D landmark detection (0-80%)
            if not reconstructor.create_xy(token=token, handle=handle):
                return None  # Cancelled

            # Stage 2: Triangulation (80-100%)
            handle.report_progress(85, "Stage 2: Triangulating 3D points")
            reconstructor.create_xyz()
            handle.report_progress(100, "Complete")

            return reconstructor

        self._processing_task = self._task_manager.submit(worker, name="reconstruction")

        # Connect signals - use QueuedConnection since TaskHandle signals
        # are emitted from worker threads
        self._processing_task.started.connect(
            self._emit_state_changed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._processing_task.completed.connect(
            self._on_reconstruction_complete,
            Qt.ConnectionType.QueuedConnection,
        )
        self._processing_task.failed.connect(
            self._on_reconstruction_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._processing_task.cancelled.connect(
            self._on_reconstruction_cancelled,
            Qt.ConnectionType.QueuedConnection,
        )
        self._processing_task.progress_updated.connect(
            self._on_progress,
            Qt.ConnectionType.QueuedConnection,
        )

        self._emit_state_changed()

    def cancel_reconstruction(self) -> None:
        """Cancel the running reconstruction task."""
        if self._processing_task is not None and self._processing_task.state == TaskState.RUNNING:
            logger.info("Cancelling reconstruction")
            self._processing_task.cancel()

    def cleanup(self) -> None:
        """Clean up resources. Call before discarding presenter."""
        self.cancel_reconstruction()

    def refresh_camera_array(self, camera_array: CameraArray) -> None:
        """Update camera array after coordinate system change.

        When the user adjusts the coordinate system origin in the calibration tab,
        the camera extrinsics change. This updates the presenter's reference and
        triggers a view rebuild if showing current calibration (not historical).
        """
        self._camera_array = camera_array
        # Only refresh if showing current calibration, not historical per-recording data
        if not self.is_showing_historical_calibration:
            self._emit_state_changed()  # Triggers view rebuild with new camera positions

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
