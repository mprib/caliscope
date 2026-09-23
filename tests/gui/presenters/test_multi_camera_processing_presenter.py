"""Tests for MultiCameraProcessingPresenter.

Canary tests for:
- State machine transitions (UNCONFIGURED -> READY -> PROCESSING -> COMPLETE)
- Processing control (start, cancel, reset)
- Key signal contracts (rotation persistence)
- Lifecycle (cleanup, config locking during processing)
"""

import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from caliscope import __root__
from caliscope.cameras.camera_array import CameraData
from caliscope.gui.presenters.multi_camera_processing_presenter import (
    MultiCameraProcessingPresenter,
    MultiCameraProcessingState,
)
from caliscope.core.process_synchronized_recording import get_initial_thumbnails
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.task_manager.task_manager import TaskManager
from caliscope.cameras.camera_array import CameraArray
from caliscope.task_manager.task_state import TaskState
from caliscope.workspace_guide import WorkspaceGuide

TEST_SESSION = Path(__root__) / "tests" / "sessions" / "4_cam_recording"


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
        0: CameraData(cam_id=0, size=(640, 480)),
        1: CameraData(cam_id=1, size=(640, 480)),
    }


@pytest.fixture
def workspace_with_recordings(tmp_path):
    """Copy test session to tmp_path for isolated testing."""
    copy_contents_to_clean_dest(TEST_SESSION, tmp_path)
    return tmp_path


@pytest.fixture
def real_camera_array(workspace_with_recordings):
    """Load real camera array from test session."""
    return CameraArray.from_toml(workspace_with_recordings / "camera_array.toml")


@pytest.fixture
def presenter(mock_task_manager, mock_tracker, tmp_path, qapp):
    """Create a MultiCameraProcessingPresenter for testing."""
    return MultiCameraProcessingPresenter(
        task_manager=mock_task_manager,
        tracker=mock_tracker,
        workspace_guide=WorkspaceGuide(tmp_path),
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
        self, minimal_cameras, workspace_with_recordings, fake_task_manager, mock_tracker, tmp_path
    ):
        """start_processing() submits task to TaskManager when READY."""
        presenter = MultiCameraProcessingPresenter(
            task_manager=fake_task_manager,
            tracker=mock_tracker,
            workspace_guide=WorkspaceGuide(tmp_path),
        )
        recording_dir = workspace_with_recordings / "recordings" / "recording_1"
        presenter.set_recording_dir(recording_dir)
        presenter.set_cameras(minimal_cameras)

        presenter.start_processing()

        assert len(fake_task_manager.handles("Multi-camera processing")) == 1

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
        presenter.rotation_changed.connect(lambda cam_id, rot: signal_received.append((cam_id, rot)))

        presenter.set_rotation(0, 1)

        assert signal_received == [(0, 1)]


class TestThumbnailLoading:
    """Thumbnail extraction from video files."""

    def test_thumbnails_load_off_the_gui_thread(
        self, workspace_with_recordings, real_camera_array, mock_tracker, qtbot, monkeypatch
    ):
        """Thumbnails are decoded in a worker thread and arrive on the GUI thread."""
        from caliscope.gui.presenters import multi_camera_processing_presenter as module

        reader_threads = []

        def recording_reader(recording_dir, cameras):
            reader_threads.append(threading.current_thread())
            return get_initial_thumbnails(recording_dir, cameras)

        monkeypatch.setattr(module, "get_initial_thumbnails", recording_reader)
        task_manager = TaskManager()
        presenter = MultiCameraProcessingPresenter(
            task_manager=task_manager,
            tracker=mock_tracker,
            workspace_guide=WorkspaceGuide(workspace_with_recordings),
        )
        cameras_dict = {cam.cam_id: cam for cam in real_camera_array.cameras.values()}

        presenter.set_recording_dir(workspace_with_recordings / "recordings" / "recording_1")
        presenter.set_cameras(cameras_dict)

        qtbot.waitUntil(lambda: len(presenter.thumbnails) == len(cameras_dict), timeout=10000)
        assert reader_threads
        assert all(thread is not threading.main_thread() for thread in reader_threads)
        assert all(isinstance(frame, np.ndarray) for frame in presenter.thumbnails.values())
        task_manager.shutdown()

    def test_thumbnails_for_a_replaced_camera_set_are_ignored(
        self, workspace_with_recordings, real_camera_array, mock_tracker, fake_task_manager, qapp
    ):
        """A thumbnail load that finishes after the camera set changed does not publish."""
        presenter = MultiCameraProcessingPresenter(
            task_manager=fake_task_manager,
            tracker=mock_tracker,
            workspace_guide=WorkspaceGuide(workspace_with_recordings),
        )
        cameras_dict = {cam.cam_id: cam for cam in real_camera_array.cameras.values()}
        presenter.set_recording_dir(workspace_with_recordings / "recordings" / "recording_1")
        presenter.set_cameras(cameras_dict)
        presenter.set_cameras({0: cameras_dict[0]})
        stale, current = fake_task_manager.handles("Load thumbnails")[-2:]

        fake_task_manager.run(stale)
        qapp.processEvents()
        assert presenter.thumbnails == {}

        fake_task_manager.run(current)
        qapp.processEvents()
        assert list(presenter.thumbnails) == [0]


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
        presenter.set_cameras({99: CameraData(cam_id=99, size=(320, 240))})

        # Original config preserved
        assert presenter.recording_dir == recording_dir
        assert 0 in presenter.cameras


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
