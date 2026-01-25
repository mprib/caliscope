"""Tests for MultiCameraProcessingPresenter.

Canary tests for:
- State machine transitions (UNCONFIGURED -> READY -> PROCESSING -> COMPLETE)
- Processing control (start, cancel, reset)
- Key signal contracts (rotation persistence)
- Lifecycle (cleanup, config locking during processing)
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

from caliscope import __root__
from caliscope.cameras.camera_array import CameraData
from caliscope.gui.presenters.multi_camera_processing_presenter import (
    MultiCameraProcessingPresenter,
    MultiCameraProcessingState,
)
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.persistence import load_camera_array
from caliscope.task_manager.task_state import TaskState

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
    """Mock task manager that returns controllable task handles."""
    return MagicMock()


@pytest.fixture
def mock_tracker():
    """Mock tracker for 2D point extraction."""
    return MagicMock()


@pytest.fixture
def minimal_cameras():
    """Create minimal CameraData objects for testing state transitions."""
    return {
        0: CameraData(port=0, size=(640, 480)),
        1: CameraData(port=1, size=(640, 480)),
    }


@pytest.fixture
def workspace_with_recordings(tmp_path):
    """Copy test session to tmp_path for isolated testing."""
    copy_contents_to_clean_dest(TEST_SESSION, tmp_path)
    return tmp_path


@pytest.fixture
def real_camera_array(workspace_with_recordings):
    """Load real camera array from test session."""
    return load_camera_array(workspace_with_recordings / "camera_array.toml")


@pytest.fixture
def presenter(mock_task_manager, mock_tracker, qapp):
    """Create a MultiCameraProcessingPresenter for testing."""
    return MultiCameraProcessingPresenter(
        task_manager=mock_task_manager,
        tracker=mock_tracker,
    )


class TestStateTransitions:
    """Core state machine behavior."""

    def test_initial_state_is_unconfigured(self, presenter):
        """New presenter without config is UNCONFIGURED."""
        assert presenter.state == MultiCameraProcessingState.UNCONFIGURED

    def test_state_becomes_ready_after_configuration(self, presenter, minimal_cameras, workspace_with_recordings):
        """State transitions to READY when both recording_dir and cameras are set."""
        recording_dir = workspace_with_recordings / "recordings" / "recording_1"
        presenter.set_recording_dir(recording_dir)
        presenter.set_cameras(minimal_cameras)

        assert presenter.state == MultiCameraProcessingState.READY

    def test_state_becomes_processing_after_start(
        self, presenter, minimal_cameras, workspace_with_recordings, mock_task_manager
    ):
        """State transitions to PROCESSING after start_processing()."""
        recording_dir = workspace_with_recordings / "recordings" / "recording_1"
        presenter.set_recording_dir(recording_dir)
        presenter.set_cameras(minimal_cameras)

        mock_handle = MagicMock()
        mock_handle.state = TaskState.RUNNING
        mock_task_manager.submit.return_value = mock_handle

        presenter.start_processing()

        assert presenter.state == MultiCameraProcessingState.PROCESSING

    def test_state_becomes_complete_after_success(self, presenter, qapp):
        """State transitions to COMPLETE when result is set."""
        presenter._result = MagicMock()
        assert presenter.state == MultiCameraProcessingState.COMPLETE


class TestProcessingControl:
    """Processing lifecycle: start, cancel, reset."""

    def test_cannot_start_processing_when_unconfigured(self, presenter, mock_task_manager):
        """start_processing() is a no-op when state is UNCONFIGURED."""
        presenter.start_processing()
        mock_task_manager.submit.assert_not_called()

    def test_start_processing_submits_task(
        self, presenter, minimal_cameras, workspace_with_recordings, mock_task_manager
    ):
        """start_processing() submits task to TaskManager when READY."""
        recording_dir = workspace_with_recordings / "recordings" / "recording_1"
        presenter.set_recording_dir(recording_dir)
        presenter.set_cameras(minimal_cameras)

        mock_handle = MagicMock()
        mock_handle.state = TaskState.RUNNING
        mock_task_manager.submit.return_value = mock_handle

        presenter.start_processing()

        mock_task_manager.submit.assert_called_once()

    def test_cancel_processing_cancels_task(
        self, presenter, minimal_cameras, workspace_with_recordings, mock_task_manager
    ):
        """cancel_processing() cancels running task."""
        recording_dir = workspace_with_recordings / "recordings" / "recording_1"
        presenter.set_recording_dir(recording_dir)
        presenter.set_cameras(minimal_cameras)

        mock_handle = MagicMock()
        mock_handle.state = TaskState.RUNNING
        mock_task_manager.submit.return_value = mock_handle

        presenter.start_processing()
        presenter.cancel_processing()

        mock_handle.cancel.assert_called_once()

    def test_reset_clears_results_but_keeps_config(self, presenter, minimal_cameras, workspace_with_recordings):
        """reset() returns to READY (not UNCONFIGURED) — config is preserved."""
        recording_dir = workspace_with_recordings / "recordings" / "recording_1"
        presenter.set_recording_dir(recording_dir)
        presenter.set_cameras(minimal_cameras)

        # Simulate completed state
        presenter._result = MagicMock()
        assert presenter.state == MultiCameraProcessingState.COMPLETE

        presenter.reset()

        assert presenter.state == MultiCameraProcessingState.READY
        assert presenter.result is None


class TestRotationControl:
    """Rotation changes emit signal for coordinator persistence."""

    def test_rotation_change_emits_signal(self, presenter, minimal_cameras, qapp):
        """set_rotation() emits rotation_changed for coordinator to persist."""
        presenter.set_cameras(minimal_cameras)

        signal_received = []
        presenter.rotation_changed.connect(lambda port, rot: signal_received.append((port, rot)))

        presenter.set_rotation(0, 1)

        assert signal_received == [(0, 1)]


class TestThumbnailLoading:
    """Thumbnail extraction from video files."""

    def test_thumbnails_loaded_on_configuration(
        self, workspace_with_recordings, real_camera_array, mock_task_manager, mock_tracker, qapp
    ):
        """Thumbnails are extracted when recording_dir and cameras are configured."""
        presenter = MultiCameraProcessingPresenter(
            task_manager=mock_task_manager,
            tracker=mock_tracker,
        )

        recording_dir = workspace_with_recordings / "recordings" / "recording_1"
        cameras_dict = {cam.port: cam for cam in real_camera_array.cameras.values()}

        presenter.set_recording_dir(recording_dir)
        presenter.set_cameras(cameras_dict)

        thumbnails = presenter.thumbnails
        assert len(thumbnails) > 0
        for frame in thumbnails.values():
            assert isinstance(frame, np.ndarray)


class TestLifecycle:
    """Cleanup and config locking during processing."""

    def test_cleanup_cancels_running_task(
        self, presenter, minimal_cameras, workspace_with_recordings, mock_task_manager
    ):
        """cleanup() cancels any running task."""
        recording_dir = workspace_with_recordings / "recordings" / "recording_1"
        presenter.set_recording_dir(recording_dir)
        presenter.set_cameras(minimal_cameras)

        mock_handle = MagicMock()
        mock_handle.state = TaskState.RUNNING
        mock_task_manager.submit.return_value = mock_handle

        presenter.start_processing()
        presenter.cleanup()

        mock_handle.cancel.assert_called_once()

    def test_config_locked_during_processing(
        self, presenter, minimal_cameras, workspace_with_recordings, mock_task_manager, qapp
    ):
        """Configuration cannot be changed while processing is active."""
        recording_dir = workspace_with_recordings / "recordings" / "recording_1"
        presenter.set_recording_dir(recording_dir)
        presenter.set_cameras(minimal_cameras)

        mock_handle = MagicMock()
        mock_handle.state = TaskState.RUNNING
        mock_task_manager.submit.return_value = mock_handle

        presenter.start_processing()

        # Attempt to change config
        presenter.set_recording_dir(Path("/other/path"))
        presenter.set_cameras({99: CameraData(port=99, size=(320, 240))})

        # Original config preserved
        assert presenter.recording_dir == recording_dir
        assert 0 in presenter.cameras


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
