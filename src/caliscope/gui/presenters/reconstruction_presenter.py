"""Presenter for reconstruction workflow (post-processing).

Coordinates the reconstruction of 3D trajectories from recorded video:
1. Stage 1: create_xy() - Synchronized 2D landmark detection
2. Stage 2: create_xyz() - Triangulation and export

This is a state machine presenter following the IntrinsicCalibrationPresenter
pattern. States are computed from reality (task state + file existence).
"""

from __future__ import annotations

import logging
import copy
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.process_synchronized_recording import process_synchronized_recording
from caliscope.core.workflow_status import WorkspaceIssue
from caliscope.gui.geometry.wireframe import WireframeSegment, wireframe_segments_from_view
from caliscope.reconstruction.reconstruct_xyz import reconstruct_xyz
from caliscope.recording.overlay_video_writer import OverlayVideoWriter
from caliscope.recording.recording_validation import (
    RecordingDimensionAssessment,
    check_recording_dimensions,
)
from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps
from caliscope.repositories.project_settings_repository import ProjectSettingsRepository
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.task_manager.task_manager import TaskManager
from caliscope.task_manager.task_state import TaskState
from caliscope.trackers import tracker_registry
from caliscope.workspace_guide import CameraVideoAssessment, WorkspaceGuide

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


class _DimensionCheckPurpose(Enum):
    """Why a selected recording's dimensions are being read."""

    SELECTION = auto()
    PROCESS = auto()


@dataclass
class _DimensionCheckRequest:
    """One background dimensions request owned by the presenter."""

    generation: int
    recording_name: str
    recording_path: Path
    expected_sizes: tuple[tuple[int, tuple[int, int] | None], ...]
    purpose: _DimensionCheckPurpose
    camera_array_snapshot: CameraArray | None = None
    handle: TaskHandle | None = None


class ReconstructionPresenter(QObject):
    """Presenter for reconstruction workflow.

    Manages the selection of recordings and trackers, submission of
    reconstruction tasks to TaskManager, and progress reporting.

    State is computed from reality:
    - Task running -> RECONSTRUCTING
    - Task failed or error set -> ERROR
    - xyz file exists -> COMPLETE
    - Otherwise -> IDLE

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
    camera_array_changed = Signal()  # camera positions changed, rebuild viz
    recordings_changed = Signal()  # recording folders or their assessments changed

    def __init__(
        self,
        workspace_dir: Path,
        workspace_guide: WorkspaceGuide,
        camera_array: CameraArray,
        task_manager: TaskManager,
        project_settings: ProjectSettingsRepository | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the presenter.

        Args:
            workspace_dir: Root workspace directory
            workspace_guide: Shared filesystem assessment gateway
            camera_array: Calibrated camera array for triangulation
            task_manager: TaskManager for background processing
            project_settings: Repository for persisting 3D view appearance settings.
            parent: Optional Qt parent
        """
        super().__init__(parent)

        self._workspace_dir = workspace_dir
        self._workspace_guide = workspace_guide
        self._camera_array = camera_array
        self._task_manager = task_manager
        self._project_settings = project_settings

        # Selection state
        self._selected_recording: str | None = None
        self._selected_tracker: str | None = None

        # Task tracking
        self._processing_task: TaskHandle | None = None
        self._last_error: str | None = None

        # Selected-recording media validation is deliberately separate from the
        # reconstruction state. It is short-lived presenter scratchpad state,
        # never a workspace-wide cache.
        self._dimension_generation = 0
        self._dimension_assessment: RecordingDimensionAssessment | None = None
        self._dimension_assessment_generation: int | None = None
        self._dimension_check_error: str | None = None
        self._dimension_check_notice: tuple[str, str] | None = None
        self._active_dimension_request: _DimensionCheckRequest | None = None
        self._pending_dimension_request: _DimensionCheckRequest | None = None
        self._dimensions_stale_during_reconstruction = False

    @property
    def state(self) -> ReconstructionState:
        """Compute current state from internal reality - never stale.

        Priority order (task state takes precedence to avoid race conditions):
        1. Task PENDING or RUNNING -> RECONSTRUCTING
        2. Task FAILED or _last_error set -> ERROR
        3. Task CANCELLED -> IDLE (cancellation returns to idle)
        4. xyz output exists -> COMPLETE
        5. Otherwise -> IDLE
        """
        if self.has_active_task:
            return ReconstructionState.RECONSTRUCTING

        if self._processing_task is not None:
            task_state = self._processing_task.state
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
        """List every immediate recording session directory, valid or not."""
        return list(self.recording_assessments)

    @property
    def recording_assessments(self) -> dict[str, CameraVideoAssessment]:
        """Current filesystem assessment for every recording session."""
        return self._workspace_guide.assess_recordings(self._camera_array.cameras)

    @property
    def recording_layout_issues(self) -> tuple[WorkspaceIssue, ...]:
        """Current issues with files and camera-split folders at recordings/."""
        return self._workspace_guide.recording_layout_issues()

    @property
    def selected_recording_assessment(self) -> CameraVideoAssessment | None:
        """Current structural assessment for the selected recording."""
        if self._selected_recording is None:
            return None
        return self.recording_assessments.get(self._selected_recording)

    @property
    def selected_recording_issues(self) -> tuple[WorkspaceIssue, ...]:
        """Current actionable issues for the selected recording."""
        if self._selected_recording is None:
            return ()
        assessment = self.selected_recording_assessment
        if assessment is not None:
            return assessment.issues
        relative_path = f"recordings/{self._selected_recording}"
        return (
            WorkspaceIssue(
                code="missing_recording_directory",
                message=f"{relative_path} no longer exists.",
                relative_path=relative_path,
            ),
        )

    @property
    def workspace_issues(self) -> tuple[WorkspaceIssue, ...]:
        """Everything the recording feedback label should say right now."""
        return (
            *self.recording_layout_issues,
            *self.selected_recording_issues,
            *self.selected_recording_dimension_issues,
        )

    @property
    def selected_recording_dimension_issues(self) -> tuple[WorkspaceIssue, ...]:
        """Dimension feedback for the selected recording's current check."""
        if self._dimension_assessment_generation != self._dimension_generation:
            return ()
        if self._dimension_check_error is not None:
            return (
                WorkspaceIssue(
                    code="recording_dimension_check_failed",
                    message=f"Could not check recording dimensions: {self._dimension_check_error}",
                    relative_path=(f"recordings/{self._selected_recording}" if self._selected_recording else None),
                ),
            )
        if self._dimension_assessment is None:
            return ()

        missing_cam_ids = set()
        assessment = self.selected_recording_assessment
        if assessment is None:
            # The structural assessment already explains that the selected
            # session disappeared. Do not add one media-read error per camera.
            return ()
        missing_cam_ids = set(assessment.missing_camera_ids)

        issues: list[WorkspaceIssue] = []
        for outcome in self._dimension_assessment.outcomes:
            if outcome.cam_id in missing_cam_ids:
                continue
            relative_path = f"recordings/{self._selected_recording}/cam_{outcome.cam_id}.mp4"
            expected = outcome.expected_size
            actual = outcome.actual_size
            if expected is None or expected[0] <= 0 or expected[1] <= 0:
                issues.append(
                    WorkspaceIssue(
                        code="missing_calibration_dimensions",
                        message=f"Calibration dimensions are unavailable for camera {outcome.cam_id}.",
                        relative_path=relative_path,
                    )
                )
            elif outcome.error is not None:
                issues.append(
                    WorkspaceIssue(
                        code="unreadable_recording_dimensions",
                        message=(
                            f"Could not read dimensions from {relative_path}. Check that the file has finished "
                            "copying and can be opened, then recheck dimensions."
                        ),
                        relative_path=relative_path,
                    )
                )
            elif actual != expected:
                assert actual is not None
                issues.append(
                    WorkspaceIssue(
                        code="recording_dimension_mismatch",
                        message=(
                            f"{relative_path} is {actual[0]}×{actual[1]}; calibration for camera "
                            f"{outcome.cam_id} expects {expected[0]}×{expected[1]}. Use a recording that "
                            "matches this calibration, or recalibrate for this recording setup."
                        ),
                        relative_path=relative_path,
                    )
                )
        return tuple(issues)

    @property
    def has_structurally_ready_selected_recording(self) -> bool:
        """Whether the selected session has the expected camera files."""
        assessment = self.selected_recording_assessment
        return bool(self._camera_array.cameras) and assessment is not None and assessment.is_ready

    @property
    def selected_recording_is_ready(self) -> bool:
        """Whether the selected session's files and dimensions are compatible."""
        return (
            self.has_structurally_ready_selected_recording
            and self._dimension_assessment_generation == self._dimension_generation
            and self._dimension_assessment is not None
            and self._dimension_assessment.is_compatible
        )

    @property
    def is_checking_dimensions(self) -> bool:
        """Whether a selected-session media check is active or queued."""
        return self._active_dimension_request is not None or self._pending_dimension_request is not None

    @property
    def dimension_check_notice(self) -> str | None:
        """A restart notice associated with the current selected session only."""
        if self._dimension_check_notice is None or self._selected_recording is None:
            return None
        recording_name, notice = self._dimension_check_notice
        return notice if recording_name == self._selected_recording else None

    @property
    def can_recheck_dimensions(self) -> bool:
        """Whether the selected session can start a manual dimensions recheck."""
        return (
            self.has_structurally_ready_selected_recording
            and not self.is_checking_dimensions
            and not self.has_active_task
        )

    @property
    def can_process(self) -> bool:
        """Whether the current recording and tracker selections are processable."""
        return (
            not self.has_active_task
            and not self.is_checking_dimensions
            and self.selected_recording_is_ready
            and self._selected_tracker is not None
        )

    @property
    def has_active_task(self) -> bool:
        """Whether reconstruction has been submitted and is not yet terminal."""
        return self._processing_task is not None and self._processing_task.state in (
            TaskState.PENDING,
            TaskState.RUNNING,
        )

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

    def _expected_sizes(self, camera_array: CameraArray) -> tuple[tuple[int, tuple[int, int] | None], ...]:
        """Return an immutable, deterministic calibration-size snapshot."""
        return tuple(sorted((cam_id, camera.size) for cam_id, camera in camera_array.cameras.items()))

    def _invalidate_dimension_validation(self, *, show_process_restart_notice: bool = False) -> None:
        """Invalidate published and queued media results without blocking the UI."""
        invalidated_process = any(
            request is not None and request.purpose == _DimensionCheckPurpose.PROCESS
            for request in (self._active_dimension_request, self._pending_dimension_request)
        )
        if show_process_restart_notice and invalidated_process and self._selected_recording is not None:
            self._dimension_check_notice = (
                self._selected_recording,
                "Recording check restarted. Select Process when it finishes.",
            )
        self._dimension_generation += 1
        self._dimension_assessment = None
        self._dimension_assessment_generation = None
        self._dimension_check_error = None
        self._pending_dimension_request = None
        if self._active_dimension_request is not None and self._active_dimension_request.handle is not None:
            self._active_dimension_request.handle.cancel()

    def _request_dimension_check(
        self,
        purpose: _DimensionCheckPurpose,
        *,
        camera_array_snapshot: CameraArray | None = None,
    ) -> None:
        """Queue a selected-session media check, retaining only the newest request."""
        if self.has_active_task:
            self._dimensions_stale_during_reconstruction = True
            return
        if not self.has_structurally_ready_selected_recording or self._selected_recording is None:
            self._notify_dimensions_changed()
            return

        camera_array = camera_array_snapshot or self._camera_array
        request = _DimensionCheckRequest(
            generation=self._dimension_generation,
            recording_name=self._selected_recording,
            recording_path=self._workspace_dir / "recordings" / self._selected_recording,
            expected_sizes=self._expected_sizes(camera_array),
            purpose=purpose,
            camera_array_snapshot=camera_array_snapshot,
        )
        if self._active_dimension_request is None:
            self._start_dimension_request(request)
        else:
            self._pending_dimension_request = request
        self._notify_dimensions_changed()

    def _start_dimension_request(self, request: _DimensionCheckRequest) -> None:
        """Submit one dimensions worker and retain ownership until its terminal signal."""

        def worker(_token, _handle):
            return check_recording_dimensions(request.recording_path, request.expected_sizes)

        handle = self._task_manager.submit(worker, name="recording-dimension-check", auto_start=False)
        request.handle = handle
        self._active_dimension_request = request
        handle.completed.connect(
            lambda result, request=request: self._on_dimension_check_completed(request, result),
            Qt.ConnectionType.QueuedConnection,
        )
        handle.failed.connect(
            lambda exc_type, message, request=request: self._on_dimension_check_failed(request, exc_type, message),
            Qt.ConnectionType.QueuedConnection,
        )
        handle.cancelled.connect(
            lambda request=request: self._on_dimension_check_cancelled(request),
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_manager.start_task(handle.task_id)

    def _request_is_current(self, request: _DimensionCheckRequest) -> bool:
        """Whether a terminal request still describes the selected inputs."""
        return (
            request.generation == self._dimension_generation
            and request.recording_name == self._selected_recording
            and request.recording_path == self._workspace_dir / "recordings" / request.recording_name
            and request.expected_sizes == self._expected_sizes(self._camera_array)
        )

    def _retire_dimension_request(self, request: _DimensionCheckRequest) -> None:
        """Release only this request's active slot, then launch the newest queued request."""
        if self._active_dimension_request is request:
            self._active_dimension_request = None
            pending = self._pending_dimension_request
            self._pending_dimension_request = None
            if pending is not None:
                self._start_dimension_request(pending)

    def _on_dimension_check_completed(self, request: _DimensionCheckRequest, result: object) -> None:
        self._retire_dimension_request(request)
        if self._request_is_current(request) and isinstance(result, RecordingDimensionAssessment):
            self._dimension_assessment = result
            self._dimension_assessment_generation = request.generation
            self._dimension_check_error = None
            if request.purpose == _DimensionCheckPurpose.PROCESS and result.is_compatible:
                self._continue_reconstruction_after_dimension_check(request)
        self._notify_dimensions_changed()

    def _on_dimension_check_failed(self, request: _DimensionCheckRequest, exc_type: str, message: str) -> None:
        self._retire_dimension_request(request)
        if self._request_is_current(request):
            self._dimension_assessment = None
            self._dimension_assessment_generation = request.generation
            self._dimension_check_error = f"{exc_type}: {message}"
        self._notify_dimensions_changed()

    def _on_dimension_check_cancelled(self, request: _DimensionCheckRequest) -> None:
        self._retire_dimension_request(request)
        self._notify_dimensions_changed()

    def _notify_dimensions_changed(self) -> None:
        """Refresh existing recording feedback and state-driven controls."""
        self.recordings_changed.emit()
        self._emit_state_changed()

    def recheck_dimensions(self) -> None:
        """Manually reread dimensions for the selected structurally valid session."""
        if not self.can_recheck_dimensions:
            return
        self._invalidate_dimension_validation()
        self._request_dimension_check(_DimensionCheckPurpose.SELECTION)

    def select_recording(self, name: str) -> None:
        """Select a recording for processing.

        Clears any previous error state when selection changes.
        """
        if self.has_active_task:
            logger.warning("Cannot change recording while reconstruction is active")
            return
        if name not in self.available_recordings:
            logger.warning(f"Recording '{name}' not in available recordings")
            return

        self._invalidate_dimension_validation()
        if name != self._selected_recording:
            self._dimension_check_notice = None
        self._selected_recording = name
        self._last_error = None  # Clear error on new selection
        self._processing_task = None  # Clear stale task reference
        self._request_dimension_check(_DimensionCheckPurpose.SELECTION)
        self._emit_state_changed()
        logger.info(f"Selected recording: {name}")

    def refresh_from_workspace(self, *, preserve_error: bool = False) -> None:
        """Reconcile selection with recording folders and emit current assessments.

        A running task keeps its original selection even if its source folder is
        removed. The task is not cancelled by filesystem feedback.
        """
        current_error = self._last_error
        assessments = self.recording_assessments
        recordings = list(assessments)
        if not self.has_active_task and self._selected_recording not in recordings:
            ready_recordings = [
                name for name, assessment in assessments.items() if self._camera_array.cameras and assessment.is_ready
            ]
            self._selected_recording = (
                ready_recordings[0] if ready_recordings else recordings[0] if recordings else None
            )
            self._last_error = None
            self._processing_task = None

        if preserve_error:
            self._last_error = current_error

        if self.has_active_task:
            self._dimensions_stale_during_reconstruction = True
        else:
            self._invalidate_dimension_validation(show_process_restart_notice=True)
            # The current selected directory may have changed in place. A
            # selection check never carries a Process continuation.
            if self._selected_recording is not None:
                self._request_dimension_check(_DimensionCheckPurpose.SELECTION)

        self.recordings_changed.emit()
        self._emit_state_changed()

    def select_tracker(self, tracker: str) -> None:
        """Select a tracker for processing.

        Clears any previous error state when selection changes.
        """
        if self.has_active_task:
            logger.warning("Cannot change tracker while reconstruction is active")
            return
        if tracker not in self.available_trackers:
            logger.warning(f"Tracker '{tracker}' not available")
            return

        self._invalidate_dimension_validation(show_process_restart_notice=True)
        self._selected_tracker = tracker
        self._last_error = None  # Clear error on new selection
        self._processing_task = None  # Clear stale task reference
        self._request_dimension_check(_DimensionCheckPurpose.SELECTION)
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

        if not self.selected_recording_is_ready:
            if not self._camera_array.cameras:
                issue_messages = ["No calibrated cameras are available for reconstruction."]
            else:
                issue_messages = [issue.message for issue in self.selected_recording_issues]
            logger.warning("Cannot start reconstruction: %s", " ".join(issue_messages))
            self._emit_state_changed()
            return

        # Check model readiness (ONNX trackers may have card but no weights)
        if not self.is_tracker_ready:
            card = tracker_registry.model_card_for(self._selected_tracker)
            if card is not None:
                self.model_download_needed.emit(card)
            return

        logger.info(f"Checking recording dimensions before reconstruction: {self._selected_recording}")
        self._last_error = None
        self._dimension_check_notice = None
        snapshot = copy.deepcopy(self._camera_array)
        self._invalidate_dimension_validation()
        self._request_dimension_check(_DimensionCheckPurpose.PROCESS, camera_array_snapshot=snapshot)

    def _continue_reconstruction_after_dimension_check(self, request: _DimensionCheckRequest) -> None:
        """Perform existing reconstruction setup after a fresh compatible preflight."""
        camera_array = request.camera_array_snapshot
        if camera_array is None or not self._request_is_current(request):
            return

        # The preflight checked headers. Reassess cheap structural facts before
        # timestamps, tracker construction, directories, or output writes.
        structural = self._workspace_guide.assess_camera_videos(request.recording_path, camera_array.cameras)
        if not structural.is_ready:
            self._invalidate_dimension_validation()
            self._request_dimension_check(_DimensionCheckPurpose.SELECTION)
            return

        recording_path = request.recording_path
        tracker_name = self._selected_tracker
        if tracker_name is None:
            return
        cam_ids = sorted(camera_array.cameras.keys())

        try:
            synced_timestamps = SynchronizedTimestamps.load(recording_path, cam_ids)
        except ValueError as e:
            # Incompatible videos: reject cleanly, stay out of RECONSTRUCTING.
            logger.error(f"Cannot load timestamps: {e}")
            self._last_error = str(e)
            self._emit_state_changed()
            return

        tracker = tracker_registry.create(tracker_name)
        tracker_dir = recording_path / tracker_name
        tracker_dir.mkdir(parents=True, exist_ok=True)
        camera_array.to_toml(tracker_dir / "camera_array.toml")

        save_overlay = bool(self._project_settings and self._project_settings.get_save_tracked_points_video())
        save_xy = bool(self._project_settings and self._project_settings.get_save_xy_points())

        def worker(token, handle):
            recorder = (
                OverlayVideoWriter(tracker_dir, tracker, synced_timestamps.mean_fps, suffix=tracker_name)
                if save_overlay
                else None
            )

            last_pct = -1

            def on_progress(done: int, total: int) -> None:
                nonlocal last_pct
                pct = int((done / total) * 80) if total else 0  # stage 1 is 0-80%
                if pct != last_pct:  # throttle: emit only on integer-percent change
                    last_pct = pct
                    raw = int((done / total) * 100) if total else 0
                    handle.report_progress(pct, f"Stage 1: {raw}% - Detecting 2D landmarks")

            def on_frame_data(sync_index, frame_data) -> None:
                if recorder is not None:
                    recorder.on_frame_data(sync_index, frame_data)

            try:
                image_points = process_synchronized_recording(
                    recording_dir=recording_path,
                    cameras=camera_array.cameras,
                    tracker=tracker,
                    synced_timestamps=synced_timestamps,
                    on_progress=on_progress,
                    on_frame_data=on_frame_data,
                    token=token,
                )
            finally:
                if recorder is not None:
                    recorder.close()  # flush encoders even on cancel/failure

            if token.is_cancelled:
                return None

            if save_xy:
                image_points.to_csv(tracker_dir / f"xy_{tracker_name}.csv")

            # Stage 2: Triangulation (80-100%), from the in-memory points (no read-back)
            handle.report_progress(85, "Stage 2: Triangulating 3D points")
            reconstruct_xyz(image_points, camera_array, tracker, tracker_dir)
            handle.report_progress(100, "Complete")

            return tracker_dir

        self._processing_task = self._task_manager.submit(worker, name="reconstruction", auto_start=False)

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
        self._task_manager.start_task(self._processing_task.task_id)

        self._emit_state_changed()

    def cancel_reconstruction(self) -> None:
        """Cancel a submitted reconstruction task."""
        if self.has_active_task and self._processing_task is not None:
            logger.info("Cancelling reconstruction")
            self._processing_task.cancel()

    # -------------------------------------------------------------------------
    # Scene Appearance Settings
    # -------------------------------------------------------------------------

    def save_camera_size_multiplier(self, multiplier: float) -> None:
        """Persist the camera frustum size multiplier to project settings."""
        if self._project_settings is not None:
            self._project_settings.set_scene_camera_size_multiplier(multiplier)

    def save_grid_size_multiplier(self, multiplier: float) -> None:
        """Persist the floor grid size multiplier to project settings."""
        if self._project_settings is not None:
            self._project_settings.set_scene_grid_size_multiplier(multiplier)

    def save_sphere_size_multiplier(self, multiplier: float) -> None:
        """Persist the point sphere size multiplier to project settings."""
        if self._project_settings is not None:
            self._project_settings.set_scene_sphere_size_multiplier(multiplier)

    def get_camera_size_multiplier(self) -> float:
        """Load camera frustum size multiplier from project settings (default: 1.0)."""
        if self._project_settings is not None:
            return self._project_settings.get_scene_camera_size_multiplier()
        return 1.0

    def get_grid_size_multiplier(self) -> float:
        """Load floor grid size multiplier from project settings (default: 1.0)."""
        if self._project_settings is not None:
            return self._project_settings.get_scene_grid_size_multiplier()
        return 1.0

    def get_sphere_size_multiplier(self) -> float:
        """Load point sphere size multiplier from project settings (default: 1.0)."""
        if self._project_settings is not None:
            return self._project_settings.get_scene_sphere_size_multiplier()
        return 1.0

    def cleanup(self) -> None:
        """Clean up resources. Call before discarding presenter."""
        self._invalidate_dimension_validation()
        self._dimension_check_notice = None
        self.cancel_reconstruction()

    def refresh_camera_array(self, camera_array: CameraArray) -> None:
        """Update camera array after coordinate system change.

        When the user adjusts the coordinate system origin in the calibration tab,
        the camera extrinsics change. This updates the presenter's reference and
        triggers a view rebuild if showing current calibration (not historical).
        """
        self._camera_array = camera_array
        self._invalidate_dimension_validation(show_process_restart_notice=True)
        self._request_dimension_check(_DimensionCheckPurpose.SELECTION)
        # Only refresh if showing current calibration, not historical per-recording data
        if not self.is_showing_historical_calibration:
            self._emit_state_changed()
            self.camera_array_changed.emit()

    def _on_reconstruction_complete(self, result: object) -> None:
        """Handle successful reconstruction."""
        assert isinstance(result, Path)
        output_path = result / f"xyz_{self._selected_tracker}.csv"
        logger.info(f"Reconstruction complete: {output_path}")

        self._emit_state_changed()

        self.reconstruction_complete.emit(output_path)
        self.refresh_from_workspace()
        self._schedule_stale_dimension_check()

    def _on_reconstruction_failed(self, exc_type: str, message: str) -> None:
        """Handle reconstruction failure."""
        error_msg = f"{exc_type}: {message}"
        logger.error(f"Reconstruction failed: {error_msg}")

        self._last_error = error_msg
        self._emit_state_changed()
        self.reconstruction_failed.emit(error_msg)
        self.refresh_from_workspace(preserve_error=True)
        self._schedule_stale_dimension_check()

    def _on_reconstruction_cancelled(self) -> None:
        """Handle reconstruction cancellation."""
        logger.info("Reconstruction was cancelled")
        self._emit_state_changed()
        self.refresh_from_workspace()
        self._schedule_stale_dimension_check()

    def _schedule_stale_dimension_check(self) -> None:
        """Run one deferred selected-session check after reconstruction ends."""
        if self._dimensions_stale_during_reconstruction:
            self._dimensions_stale_during_reconstruction = False
            if not self.is_checking_dimensions:
                self._invalidate_dimension_validation()
                self._request_dimension_check(_DimensionCheckPurpose.SELECTION)

    def _on_progress(self, percent: int, message: str) -> None:
        """Forward progress updates to our signal."""
        self.progress_updated.emit(percent, message)

    def _emit_state_changed(self) -> None:
        """Emit state_changed signal with current computed state."""
        current_state = self.state
        logger.debug(f"State changed to {current_state}")
        self.state_changed.emit(current_state)
