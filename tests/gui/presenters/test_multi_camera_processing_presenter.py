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
import pandas as pd
import pytest

from caliscope import __root__
from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.point_data import ImagePoints
from caliscope.core.process_synchronized_recording import get_initial_thumbnails
from caliscope.gui.presenters.multi_camera_processing_presenter import (
    MultiCameraProcessingPresenter,
    MultiCameraProcessingState,
)
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.task_manager.task_manager import TaskManager
from caliscope.workspace_guide import WorkspaceGuide

TEST_SESSION = Path(__root__) / "tests" / "sessions" / "4_cam_recording"
PROCESSING = "Multi-camera processing"


@pytest.fixture
def minimal_cameras():
    return {
        0: CameraData(cam_id=0, size=(640, 480)),
        1: CameraData(cam_id=1, size=(640, 480)),
    }


@pytest.fixture
def workspace(tmp_path):
    copy_contents_to_clean_dest(TEST_SESSION, tmp_path)
    return tmp_path


@pytest.fixture
def recording_dir(workspace):
    return workspace / "recordings" / "recording_1"


@pytest.fixture
def real_cameras(workspace):
    return dict(CameraArray.from_toml(workspace / "camera_array.toml").cameras)


@pytest.fixture
def presenter(fake_task_manager, workspace):
    return MultiCameraProcessingPresenter(
        task_manager=fake_task_manager,
        tracker=MagicMock(),
        workspace_guide=WorkspaceGuide(workspace),
    )


@pytest.fixture
def ready_presenter(presenter, minimal_cameras, recording_dir):
    presenter.set_recording_dir(recording_dir)
    presenter.set_cameras(minimal_cameras)
    return presenter


def _image_points() -> ImagePoints:
    return ImagePoints(
        pd.DataFrame(
            {
                "sync_index": [0, 0],
                "cam_id": [0, 1],
                "object_id": [0, 0],
                "keypoint_id": [0, 0],
                "img_loc_x": [1.0, 2.0],
                "img_loc_y": [3.0, 4.0],
            }
        )
    )


class TestStateTransitions:
    def test_initial_state_is_unconfigured(self, presenter):
        assert presenter.state == MultiCameraProcessingState.UNCONFIGURED

    def test_state_becomes_ready_after_configuration(self, ready_presenter):
        assert ready_presenter.state == MultiCameraProcessingState.READY

    def test_start_processing_submits_task_and_enters_processing(self, ready_presenter, fake_task_manager):
        ready_presenter.start_processing()

        assert len(fake_task_manager.handles(PROCESSING)) == 1
        assert ready_presenter.state == MultiCameraProcessingState.PROCESSING

    def test_state_becomes_complete_after_success(self, ready_presenter, fake_task_manager, qapp):
        ready_presenter.start_processing()
        (handle,) = fake_task_manager.handles(PROCESSING)

        fake_task_manager.complete(handle, _image_points())
        qapp.processEvents()

        assert ready_presenter.state == MultiCameraProcessingState.COMPLETE


class TestProcessingControl:
    def test_cannot_start_processing_when_unconfigured(self, presenter, fake_task_manager):
        presenter.start_processing()
        assert fake_task_manager.submitted == []

    @pytest.mark.parametrize("stop", ["cancel_processing", "cleanup"])
    def test_stopping_cancels_running_task(self, ready_presenter, fake_task_manager, stop):
        ready_presenter.start_processing()
        (handle,) = fake_task_manager.handles(PROCESSING)
        fake_task_manager.start(handle)

        getattr(ready_presenter, stop)()

        assert fake_task_manager.cancel_requested(handle)

    def test_reset_clears_results_but_keeps_config(self, ready_presenter, fake_task_manager, qapp):
        """reset() returns to READY (not UNCONFIGURED): config is preserved."""
        ready_presenter.start_processing()
        fake_task_manager.complete(fake_task_manager.handles(PROCESSING)[0], _image_points())
        qapp.processEvents()
        assert ready_presenter.state == MultiCameraProcessingState.COMPLETE

        ready_presenter.reset()

        assert ready_presenter.state == MultiCameraProcessingState.READY
        assert ready_presenter.result is None


class TestRotationControl:
    def test_rotation_change_emits_signal(self, presenter, minimal_cameras):
        """set_rotation() emits rotation_changed for coordinator to persist."""
        presenter.set_cameras(minimal_cameras)
        received = []
        presenter.rotation_changed.connect(lambda cam_id, rot: received.append((cam_id, rot)))

        presenter.set_rotation(0, 1)

        assert received == [(0, 1)]


class TestThumbnailLoading:
    def test_thumbnails_load_off_the_gui_thread(self, workspace, recording_dir, real_cameras, qtbot, monkeypatch):
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
            tracker=MagicMock(),
            workspace_guide=WorkspaceGuide(workspace),
        )

        presenter.set_recording_dir(recording_dir)
        presenter.set_cameras(real_cameras)

        qtbot.waitUntil(lambda: len(presenter.thumbnails) == len(real_cameras), timeout=10000)
        assert reader_threads
        assert all(thread is not threading.main_thread() for thread in reader_threads)
        assert all(isinstance(frame, np.ndarray) for frame in presenter.thumbnails.values())
        task_manager.shutdown()

    def test_thumbnails_for_a_replaced_camera_set_are_ignored(
        self, presenter, recording_dir, real_cameras, fake_task_manager, qapp
    ):
        """A thumbnail load that finishes after the camera set changed does not publish."""
        presenter.set_recording_dir(recording_dir)
        presenter.set_cameras(real_cameras)
        presenter.set_cameras({0: real_cameras[0]})
        stale, current = fake_task_manager.handles("Load thumbnails")[-2:]

        fake_task_manager.run(stale)
        qapp.processEvents()
        assert presenter.thumbnails == {}

        fake_task_manager.run(current)
        qapp.processEvents()
        assert list(presenter.thumbnails) == [0]


class TestLifecycle:
    def test_config_locked_during_processing(self, ready_presenter, recording_dir):
        """Configuration cannot be changed while processing is active."""
        ready_presenter.start_processing()

        ready_presenter.set_recording_dir(Path("/other/path"))
        ready_presenter.set_cameras({99: CameraData(cam_id=99, size=(320, 240))})

        assert ready_presenter.recording_dir == recording_dir
        assert 0 in ready_presenter.cameras
