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
from PySide6.QtCore import QCoreApplication

from caliscope import __root__
from caliscope.gui.presenters.reconstruction_presenter import (
    ReconstructionPresenter,
    ReconstructionState,
)
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.persistence import load_camera_array
from caliscope.task_manager.task_state import TaskState
from caliscope.trackers.tracker_enum import TrackerEnum

# Test session with 4 cameras and recordings
TEST_SESSION = Path(__root__) / "tests" / "sessions" / "4_cam_recording"


@pytest.fixture
def qapp():
    """Ensure QCoreApplication exists for Qt signal tests."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    yield app


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
    return load_camera_array(workspace_with_recordings / "camera_array.toml")


@pytest.fixture
def presenter(workspace_with_recordings, camera_array, mock_task_manager, qapp):
    """Create a ReconstructionPresenter for testing."""
    return ReconstructionPresenter(
        workspace_dir=workspace_with_recordings,
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
        """State is COMPLETE when xyz output file exists."""
        presenter._selected_recording = "recording_1"
        presenter._selected_tracker = TrackerEnum.SIMPLE_HOLISTIC

        # Create the output file (exist_ok because test session may have partial data)
        output_dir = workspace_with_recordings / "recordings" / "recording_1" / "SIMPLE_HOLISTIC"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "xyz_SIMPLE_HOLISTIC.csv").touch()

        assert presenter.state == ReconstructionState.COMPLETE

    def test_task_state_takes_precedence_over_file(self, presenter, workspace_with_recordings):
        """Task RUNNING state takes precedence over file existence."""
        presenter._selected_recording = "recording_1"
        presenter._selected_tracker = TrackerEnum.SIMPLE_HOLISTIC

        # Create the output file (exist_ok because test session may have partial data)
        output_dir = workspace_with_recordings / "recordings" / "recording_1" / "SIMPLE_HOLISTIC"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "xyz_SIMPLE_HOLISTIC.csv").touch()

        # But task is running
        mock_task = MagicMock()
        mock_task.state = TaskState.RUNNING
        presenter._processing_task = mock_task

        # Task state should win
        assert presenter.state == ReconstructionState.RECONSTRUCTING


class TestAvailableOptions:
    """Tests for available recordings and trackers."""

    def test_available_recordings(self, presenter):
        """Should list valid recording directories."""
        recordings = presenter.available_recordings
        assert "recording_1" in recordings
        assert len(recordings) == 1  # Test session has one recording

    def test_available_trackers_excludes_charuco(self, presenter):
        """CHARUCO should not be in available trackers."""
        trackers = presenter.available_trackers
        tracker_names = [t.name for t in trackers]

        assert "CHARUCO" not in tracker_names
        assert "SIMPLE_HOLISTIC" in tracker_names


class TestSelection:
    """Tests for recording and tracker selection."""

    def test_select_recording(self, presenter):
        """Selecting a recording updates selection."""
        presenter.select_recording("recording_1")
        assert presenter.selected_recording == "recording_1"

    def test_select_tracker(self, presenter):
        """Selecting a tracker updates selection."""
        presenter.select_tracker(TrackerEnum.SIMPLE_HOLISTIC)
        assert presenter.selected_tracker == TrackerEnum.SIMPLE_HOLISTIC

    def test_select_recording_clears_error(self, presenter):
        """Selecting a recording clears previous error."""
        presenter._last_error = "Previous error"
        presenter.select_recording("recording_1")
        assert presenter._last_error is None

    def test_select_tracker_clears_error(self, presenter):
        """Selecting a tracker clears previous error."""
        presenter._last_error = "Previous error"
        presenter.select_tracker(TrackerEnum.SIMPLE_HOLISTIC)
        assert presenter._last_error is None

    def test_select_invalid_recording_ignored(self, presenter):
        """Selecting an invalid recording is ignored."""
        presenter.select_recording("nonexistent")
        assert presenter.selected_recording is None


class TestXyzOutputPath:
    """Tests for xyz_output_path computation."""

    def test_xyz_output_path_none_when_no_selection(self, presenter):
        """Path is None when recording or tracker not selected."""
        assert presenter.xyz_output_path is None

    def test_xyz_output_path_computed_from_selection(self, presenter, workspace_with_recordings):
        """Path is computed from selected recording and tracker."""
        presenter._selected_recording = "recording_1"
        presenter._selected_tracker = TrackerEnum.SIMPLE_HOLISTIC

        expected = (
            workspace_with_recordings / "recordings" / "recording_1" / "SIMPLE_HOLISTIC" / "xyz_SIMPLE_HOLISTIC.csv"
        )
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

    def test_state_changed_emitted_on_tracker_selection(self, presenter, qapp):
        """state_changed signal emitted when tracker selection changes."""
        signal_received = []
        presenter.state_changed.connect(lambda s: signal_received.append(s))

        presenter.select_tracker(TrackerEnum.SIMPLE_HOLISTIC)

        assert len(signal_received) == 1


class TestStartReconstruction:
    """Tests for starting reconstruction."""

    def test_cannot_start_without_selection(self, presenter, qapp):
        """Cannot start reconstruction without both selections."""
        presenter.start_reconstruction()

        # Should set error and not submit task
        assert presenter._last_error is not None
        presenter._task_manager.submit.assert_not_called()

    def test_start_submits_task(self, presenter, mock_task_manager, qapp):
        """Starting reconstruction submits task to manager."""
        mock_handle = MagicMock()
        mock_handle.state = TaskState.RUNNING
        mock_task_manager.submit.return_value = mock_handle

        presenter.select_recording("recording_1")
        presenter.select_tracker(TrackerEnum.SIMPLE_HOLISTIC)
        presenter.start_reconstruction()

        mock_task_manager.submit.assert_called_once()
        call_kwargs = mock_task_manager.submit.call_args
        assert call_kwargs.kwargs["name"] == "reconstruction"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
