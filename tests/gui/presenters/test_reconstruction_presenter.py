"""Tests for ReconstructionPresenter.

Tests focus on:
- State computation from mocked file existence and task states
- Signal emissions on state transitions
- Selection clears error state
- cleanup() cancels active task
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from caliscope import __root__
from caliscope.gui.presenters.reconstruction_presenter import (
    ReconstructionPresenter,
    ReconstructionState,
)
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.cameras.camera_array import CameraArray
from caliscope.task_manager.task_state import TaskState
from caliscope.trackers import tracker_registry
from caliscope.workspace_guide import WorkspaceGuide

# Test session with 4 cameras and recordings
TEST_SESSION = Path(__root__) / "tests" / "sessions" / "4_cam_recording"
_CHARUCO_SESSION = Path(__root__) / "tests" / "sessions" / "post_optimization"

# Tracker key used throughout this test module
_TEST_TRACKER = "CHARUCO"


def _create_recording_session(workspace: Path, name: str) -> Path:
    session = workspace / "recordings" / name
    session.mkdir()
    for cam_id in (0, 1, 2, 3):
        (session / f"cam_{cam_id}.mp4").touch()
    return session


def _remove_recording_session(session: Path) -> None:
    for child in session.iterdir():
        child.unlink()
    session.rmdir()


@pytest.fixture
def registered_test_tracker():
    """Register a real CharucoTracker under the test tracker key for the duration of a test.

    The reconstruction presenter uses tracker_registry.available_names() to determine
    which trackers can be selected. This fixture ensures the test tracker key is registered
    so presenter selection methods work correctly.
    """
    from caliscope.core.charuco import Charuco
    from caliscope.trackers.charuco_tracker import CharucoTracker

    charuco = Charuco.from_toml(_CHARUCO_SESSION / "charuco.toml")
    tracker_registry.register(_TEST_TRACKER, lambda: CharucoTracker(charuco), display_name="Charuco")
    yield
    # Remove just the test key to avoid polluting other tests
    tracker_registry._factories.pop(_TEST_TRACKER, None)
    tracker_registry._display_names.pop(_TEST_TRACKER, None)
    tracker_registry._wireframes.pop(_TEST_TRACKER, None)


@pytest.fixture
def mock_task_manager():
    """Mock task manager."""
    return MagicMock()


@pytest.fixture
def workspace_with_recordings(tmp_path):
    """Copy test session to tmp_path for isolated testing."""
    copy_contents_to_clean_dest(TEST_SESSION, tmp_path)
    return tmp_path


@pytest.fixture
def camera_array(workspace_with_recordings):
    """Load real camera array from test session."""
    return CameraArray.from_toml(workspace_with_recordings / "camera_array.toml")


@pytest.fixture
def presenter(workspace_with_recordings, camera_array, mock_task_manager, qapp):
    """Create a ReconstructionPresenter for testing."""
    return ReconstructionPresenter(
        workspace_dir=workspace_with_recordings,
        workspace_guide=WorkspaceGuide(workspace_with_recordings),
        camera_array=camera_array,
        task_manager=mock_task_manager,
    )


class TestStateComputation:
    """Tests for state computation from reality."""

    def test_initial_state_is_idle(self, presenter):
        """Presenter starts in IDLE state."""
        assert presenter.state == ReconstructionState.IDLE

    def test_state_reconstructing_when_task_running(self, presenter):
        """State is RECONSTRUCTING when task is running."""
        mock_task = MagicMock()
        mock_task.state = TaskState.RUNNING
        presenter._processing_task = mock_task

        assert presenter.state == ReconstructionState.RECONSTRUCTING

    def test_state_reconstructing_when_task_pending(self, presenter):
        """A submitted task is active before its worker thread starts."""
        mock_task = MagicMock()
        mock_task.state = TaskState.PENDING
        presenter._processing_task = mock_task

        assert presenter.has_active_task is True
        assert presenter.state == ReconstructionState.RECONSTRUCTING

    def test_state_error_when_task_failed(self, presenter):
        """State is ERROR when task has failed."""
        mock_task = MagicMock()
        mock_task.state = TaskState.FAILED
        presenter._processing_task = mock_task

        assert presenter.state == ReconstructionState.ERROR

    def test_state_error_when_last_error_set(self, presenter):
        """State is ERROR when _last_error is set."""
        presenter._last_error = "Something went wrong"

        assert presenter.state == ReconstructionState.ERROR

    def test_state_idle_when_task_cancelled(self, presenter):
        """State returns to IDLE when task is cancelled."""
        mock_task = MagicMock()
        mock_task.state = TaskState.CANCELLED
        presenter._processing_task = mock_task

        assert presenter.state == ReconstructionState.IDLE

    def test_state_complete_when_xyz_exists(self, presenter, workspace_with_recordings):
        """Existing output remains complete even if its source recording degrades."""
        presenter._selected_recording = "recording_1"
        presenter._selected_tracker = "CHARUCO"

        # Create the output file (exist_ok because test session may have partial data)
        output_dir = workspace_with_recordings / "recordings" / "recording_1" / "CHARUCO"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "xyz_CHARUCO.csv").touch()
        (workspace_with_recordings / "recordings" / "recording_1" / "cam_2.mp4").unlink()

        assert presenter.state == ReconstructionState.COMPLETE
        assert presenter.selected_recording_is_ready is False

    def test_task_state_takes_precedence_over_file(self, presenter, workspace_with_recordings):
        """Task RUNNING state takes precedence over file existence."""
        presenter._selected_recording = "recording_1"
        presenter._selected_tracker = "CHARUCO"

        # Create the output file (exist_ok because test session may have partial data)
        output_dir = workspace_with_recordings / "recordings" / "recording_1" / "CHARUCO"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "xyz_CHARUCO.csv").touch()

        # But task is running
        mock_task = MagicMock()
        mock_task.state = TaskState.RUNNING
        presenter._processing_task = mock_task

        # Task state should win
        assert presenter.state == ReconstructionState.RECONSTRUCTING


class TestAvailableOptions:
    """Tests for available recordings and trackers."""

    def test_available_recordings_includes_invalid_session_directories(self, presenter, workspace_with_recordings):
        """The list mirrors immediate session directories instead of filtering them."""
        (workspace_with_recordings / "recordings" / "empty_session").mkdir()

        recordings = presenter.available_recordings

        assert "recording_1" in recordings
        assert "empty_session" in recordings

    def test_available_trackers_excludes_charuco(self, presenter):
        """CHARUCO is a calibration tracker and must not appear in reconstruction trackers."""
        trackers = presenter.available_trackers

        assert "CHARUCO" not in trackers


class TestSelection:
    """Tests for recording and tracker selection."""

    def test_select_recording(self, presenter):
        """Selecting a recording updates selection."""
        presenter.select_recording("recording_1")
        assert presenter.selected_recording == "recording_1"

    def test_select_tracker(self, presenter, registered_test_tracker):
        """Selecting a tracker updates selection."""
        presenter.select_tracker("CHARUCO")
        assert presenter.selected_tracker == "CHARUCO"

    def test_select_recording_clears_error(self, presenter):
        """Selecting a recording clears previous error."""
        presenter._last_error = "Previous error"
        presenter.select_recording("recording_1")
        assert presenter._last_error is None

    def test_select_tracker_clears_error(self, presenter, registered_test_tracker):
        """Selecting a tracker clears previous error."""
        presenter._last_error = "Previous error"
        presenter.select_tracker("CHARUCO")
        assert presenter._last_error is None

    def test_select_structurally_invalid_recording_shows_its_assessment(
        self,
        presenter,
        workspace_with_recordings,
    ):
        """Invalid session rows remain selectable so their feedback is visible."""
        invalid = workspace_with_recordings / "recordings" / "partial"
        invalid.mkdir()
        (invalid / "cam_0.mp4").touch()

        presenter.select_recording("partial")

        assert presenter.selected_recording == "partial"
        assert presenter.selected_recording_is_ready is False
        assert presenter.selected_recording_assessment is not None
        assert presenter.selected_recording_assessment.missing_camera_ids == (1, 2, 3)

    def test_nonexistent_recording_is_ignored(self, presenter):
        presenter.select_recording("nonexistent")
        assert presenter.selected_recording is None

    def test_refresh_preserves_selection_and_recovers_after_video_restore(
        self,
        presenter,
        workspace_with_recordings,
        registered_test_tracker,
    ):
        recording = workspace_with_recordings / "recordings" / "recording_1"
        presenter.select_recording("recording_1")
        presenter.select_tracker("CHARUCO")

        (recording / "cam_2.mp4").unlink()
        presenter.refresh_recordings()

        assert presenter.selected_recording == "recording_1"
        assert presenter.can_process is False
        assert [issue.relative_path for issue in presenter.selected_recording_assessment.issues] == [
            "recordings/recording_1/cam_2.mp4"
        ]

        (recording / "cam_2.mp4").touch()
        presenter.refresh_recordings()

        assert presenter.selected_recording == "recording_1"
        assert presenter.can_process is True

    @pytest.mark.parametrize("task_state", [TaskState.PENDING, TaskState.RUNNING])
    def test_refresh_reports_removed_active_session_without_cancelling(
        self,
        presenter,
        workspace_with_recordings,
        task_state,
    ):
        recording = workspace_with_recordings / "recordings" / "recording_1"
        presenter.select_recording("recording_1")
        active_task = MagicMock()
        active_task.state = task_state
        presenter._processing_task = active_task
        _remove_recording_session(recording)

        presenter.refresh_recordings()

        assert presenter.selected_recording == "recording_1"
        assert [issue.code for issue in presenter.selected_recording_issues] == ["missing_recording_directory"]
        active_task.cancel.assert_not_called()


class TestXyzOutputPath:
    """Tests for xyz_output_path computation."""

    def test_xyz_output_path_none_when_no_selection(self, presenter):
        """Path is None when recording or tracker not selected."""
        assert presenter.xyz_output_path is None

    def test_xyz_output_path_computed_from_selection(self, presenter, workspace_with_recordings):
        """Path is computed from selected recording and tracker."""
        presenter._selected_recording = "recording_1"
        presenter._selected_tracker = "CHARUCO"

        expected = workspace_with_recordings / "recordings" / "recording_1" / "CHARUCO" / "xyz_CHARUCO.csv"
        assert presenter.xyz_output_path == expected


class TestSignalEmissions:
    """Tests for signal emissions."""

    def test_state_changed_emitted_on_selection(self, presenter, qapp):
        """state_changed signal emitted when selection changes."""
        signal_received = []
        presenter.state_changed.connect(lambda s: signal_received.append(s))

        presenter.select_recording("recording_1")

        assert len(signal_received) == 1
        assert signal_received[0] == ReconstructionState.IDLE

    def test_state_changed_emitted_on_tracker_selection(self, presenter, qapp, registered_test_tracker):
        """state_changed signal emitted when tracker selection changes."""
        signal_received = []
        presenter.state_changed.connect(lambda s: signal_received.append(s))

        presenter.select_tracker("CHARUCO")

        assert len(signal_received) == 1


class TestStartReconstruction:
    """Tests for starting reconstruction."""

    def test_cannot_start_without_selection(self, presenter, qapp):
        """Cannot start reconstruction without both selections."""
        presenter.start_reconstruction()

        # Should set error and not submit task
        assert presenter._last_error is not None
        presenter._task_manager.submit.assert_not_called()

    def test_start_submits_task(self, presenter, mock_task_manager, qapp, registered_test_tracker):
        """Starting reconstruction submits task to manager."""
        mock_handle = MagicMock()
        mock_handle.state = TaskState.RUNNING
        mock_task_manager.submit.return_value = mock_handle

        presenter.select_recording("recording_1")
        presenter.select_tracker("CHARUCO")
        presenter.start_reconstruction()

        mock_task_manager.submit.assert_called_once()
        call_kwargs = mock_task_manager.submit.call_args
        assert call_kwargs.kwargs["name"] == "reconstruction"

    def test_missing_camera_video_blocks_submission_even_with_timestamps(
        self,
        presenter,
        workspace_with_recordings,
        mock_task_manager,
        qapp,
        registered_test_tracker,
    ):
        recording = workspace_with_recordings / "recordings" / "recording_1"
        assert (recording / "timestamps.csv").exists()
        (recording / "cam_3.mp4").unlink()
        presenter.select_recording("recording_1")
        presenter.select_tracker("CHARUCO")

        presenter.start_reconstruction()

        mock_task_manager.submit.assert_not_called()
        assert presenter.selected_recording_is_ready is False

    def test_timestamp_failure_creates_no_tracker_output_or_task(
        self,
        presenter,
        workspace_with_recordings,
        mock_task_manager,
        qapp,
        registered_test_tracker,
        monkeypatch,
    ):
        recording = workspace_with_recordings / "recordings" / "recording_1"
        load_timestamps = MagicMock(side_effect=ValueError("Timestamp data is incompatible"))
        create_tracker = MagicMock()
        monkeypatch.setattr(
            "caliscope.gui.presenters.reconstruction_presenter.SynchronizedTimestamps.load",
            load_timestamps,
        )
        monkeypatch.setattr(tracker_registry, "create", create_tracker)
        presenter.select_recording("recording_1")
        presenter.select_tracker("CHARUCO")

        presenter.start_reconstruction()

        load_timestamps.assert_called_once()
        create_tracker.assert_not_called()
        assert not (recording / "CHARUCO").exists()
        mock_task_manager.submit.assert_not_called()

    def test_repeated_start_does_not_submit_while_first_task_is_pending(
        self,
        presenter,
        mock_task_manager,
        qapp,
        registered_test_tracker,
    ):
        pending_handle = MagicMock()
        pending_handle.state = TaskState.PENDING
        mock_task_manager.submit.return_value = pending_handle
        presenter.select_recording("recording_1")
        presenter.select_tracker("CHARUCO")

        presenter.start_reconstruction()
        presenter.start_reconstruction()

        mock_task_manager.submit.assert_called_once()
        mock_task_manager.start_task.assert_called_once()


class TestTerminalSelectionReconciliation:
    def test_completion_emits_original_result_before_selecting_remaining_session(
        self,
        presenter,
        workspace_with_recordings,
        registered_test_tracker,
    ):
        remaining = _create_recording_session(workspace_with_recordings, "remaining")
        original = workspace_with_recordings / "recordings" / "recording_1"
        presenter.select_recording("recording_1")
        presenter.select_tracker("CHARUCO")
        completed_task = MagicMock()
        completed_task.state = TaskState.COMPLETED
        presenter._processing_task = completed_task
        _remove_recording_session(original)
        result_dir = workspace_with_recordings / "completed-result"
        observed: list[tuple[Path, str | None]] = []
        presenter.reconstruction_complete.connect(lambda path: observed.append((path, presenter.selected_recording)))

        presenter._on_reconstruction_complete(result_dir)

        assert observed == [(result_dir / "xyz_CHARUCO.csv", "recording_1")]
        assert presenter.selected_recording == remaining.name

    def test_failure_reconciles_removed_selection_after_failure_signal(
        self,
        presenter,
        workspace_with_recordings,
    ):
        remaining = _create_recording_session(workspace_with_recordings, "remaining")
        original = workspace_with_recordings / "recordings" / "recording_1"
        presenter.select_recording("recording_1")
        failed_task = MagicMock()
        failed_task.state = TaskState.FAILED
        presenter._processing_task = failed_task
        _remove_recording_session(original)
        observed: list[tuple[str, str | None]] = []
        presenter.reconstruction_failed.connect(
            lambda message: observed.append((message, presenter.selected_recording))
        )

        presenter._on_reconstruction_failed("RuntimeError", "worker failed")

        assert observed == [("RuntimeError: worker failed", "recording_1")]
        assert presenter.selected_recording == remaining.name
        assert presenter.last_error == "RuntimeError: worker failed"
        assert presenter.state == ReconstructionState.ERROR

    def test_cancellation_reconciles_removed_selection(self, presenter, workspace_with_recordings):
        remaining = _create_recording_session(workspace_with_recordings, "remaining")
        original = workspace_with_recordings / "recordings" / "recording_1"
        presenter.select_recording("recording_1")
        cancelled_task = MagicMock()
        cancelled_task.state = TaskState.CANCELLED
        presenter._processing_task = cancelled_task
        _remove_recording_session(original)

        presenter._on_reconstruction_cancelled()

        assert presenter.selected_recording == remaining.name
        assert presenter.state == ReconstructionState.IDLE


class TestCleanup:
    """Tests for cleanup behavior."""

    def test_cleanup_cancels_running_task(self, presenter):
        """cleanup() cancels any running task."""
        mock_task = MagicMock()
        mock_task.state = TaskState.RUNNING
        presenter._processing_task = mock_task

        presenter.cleanup()

        mock_task.cancel.assert_called_once()

    def test_cleanup_safe_when_no_task(self, presenter):
        """cleanup() is safe to call when no task exists."""
        presenter.cleanup()  # Should not raise

    def test_cancel_accepts_pending_task(self, presenter):
        """A task can be cancelled before its worker thread begins."""
        mock_task = MagicMock()
        mock_task.state = TaskState.PENDING
        presenter._processing_task = mock_task

        presenter.cancel_reconstruction()

        mock_task.cancel.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
