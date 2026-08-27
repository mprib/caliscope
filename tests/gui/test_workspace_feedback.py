"""Workspace file feedback reaching already-built views."""

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QElapsedTimer, QEventLoop
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QLabel, QListWidget, QPushButton

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.gui.multi_camera_processing_tab import MultiCameraProcessingTab
from caliscope.gui.presenters.multi_camera_processing_presenter import MultiCameraProcessingState
from caliscope.gui.reconstruction_tab import ReconstructionTab
from caliscope.gui.views.project_setup_view import ProjectSetupView
from caliscope.trackers import tracker_registry
from caliscope.workspace_coordinator import WorkspaceCoordinator


def _wait_until(
    qapp: QApplication,
    condition: Callable[[], bool],
    timeout_ms: int = 3000,
) -> None:
    """Process Qt events until a filesystem-driven UI condition becomes true."""
    timer = QElapsedTimer()
    timer.start()
    while not condition():
        qapp.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if timer.elapsed() >= timeout_ms:
            raise AssertionError("Timed out waiting for filesystem-driven UI update")


def _wait_for_status_change(
    qapp: QApplication,
    spy: QSignalSpy,
    previous_count: int,
    condition: Callable[[], bool],
) -> None:
    """Wait for both Coordinator notification and its rendered effect."""
    _wait_until(qapp, lambda: spy.count() > previous_count and condition())


@pytest.fixture
def coordinator(tmp_path: Path, qapp, monkeypatch: pytest.MonkeyPatch) -> Iterator[WorkspaceCoordinator]:
    monkeypatch.setattr(
        "caliscope.workspace_coordinator.read_video_properties",
        lambda _path: {"size": (640, 480)},
    )
    coordinator = WorkspaceCoordinator(tmp_path)
    yield coordinator
    coordinator.cleanup()


def test_project_feedback_follows_directory_change(coordinator: WorkspaceCoordinator) -> None:
    extrinsic = coordinator.workspace_guide.extrinsic_dir
    view = ProjectSetupView(coordinator)

    assert "No extrinsic camera videos found" in view._file_feedback_label.text()

    (extrinsic / "cam_0.mp4").touch()
    coordinator._on_directory_changed(str(extrinsic))

    assert "No extrinsic camera videos found" not in view._file_feedback_label.text()
    assert tuple(coordinator.camera_array.cameras) == (0,)


def test_open_extract_tab_follows_extrinsic_videos(coordinator: WorkspaceCoordinator) -> None:
    extrinsic = coordinator.workspace_guide.extrinsic_dir
    for cam_id in (0, 1):
        (extrinsic / f"cam_{cam_id}.mp4").touch()
    coordinator.load_camera_array()
    tab = MultiCameraProcessingTab(coordinator)
    assert tab._presenter is not None and tab._widget is not None
    assert tab._file_warning_label.isHidden()
    assert sorted(tab._widget._camera_cards) == [0, 1]

    (extrinsic / "cam_1.mp4").unlink()
    coordinator._on_directory_changed(str(extrinsic))

    assert "Missing calibration/extrinsic/cam_1.mp4" in tab._file_warning_label.text()
    assert tuple(coordinator.camera_array.cameras) == (0, 1)
    assert tab._presenter.state == MultiCameraProcessingState.UNCONFIGURED
    assert tab._widget._camera_cards[1]._thumbnail_label.text() == "No video file for cam_1"
    assert not tab._widget._action_btn.isEnabled()

    (extrinsic / "cam_1.mp4").touch()
    (extrinsic / "cam_2.mp4").touch()
    coordinator._on_directory_changed(str(extrinsic))

    assert tab._file_warning_label.isHidden()
    assert tab._presenter.state == MultiCameraProcessingState.READY
    assert tab._widget._action_btn.isEnabled()
    assert sorted(tab._widget._camera_cards) == [0, 1, 2]
    tab.cleanup()


def test_reconstruction_tab_follows_nested_recording_changes_through_real_watcher(
    tmp_path: Path,
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A built tab follows nested camera files through QFileSystemWatcher."""
    tracker_name = "TEST_FEEDBACK"
    tracker_registry.register(tracker_name, lambda: MagicMock(), display_name="Test Feedback")
    monkeypatch.setattr("caliscope.gui.views.reconstruction_widget.opengl_available", lambda: False)
    monkeypatch.chdir(tmp_path.parent)
    relative_workspace = Path(tmp_path.name)
    coordinator = WorkspaceCoordinator(relative_workspace)
    coordinator.camera_array = CameraArray(
        {
            0: CameraData(cam_id=0, size=(640, 480)),
            1: CameraData(cam_id=1, size=(640, 480)),
        }
    )
    presenter = coordinator.create_reconstruction_presenter()
    tab = ReconstructionTab(presenter, coordinator)
    status_spy = QSignalSpy(coordinator.status_changed)

    recording_list = tab.findChild(QListWidget, "recordingList")
    feedback_label = tab.findChild(QLabel, "recordingFeedbackLabel")
    process_button = tab.findChild(QPushButton, "processButton")
    assert recording_list is not None
    assert feedback_label is not None
    assert process_button is not None
    presenter.select_tracker(tracker_name)

    try:
        session = coordinator.workspace_guide.recording_dir / "walk"
        previous_count = status_spy.count()
        session.mkdir()
        _wait_for_status_change(
            qapp,
            status_spy,
            previous_count,
            lambda: (
                str(session.resolve()) in coordinator._watcher.directories()
                and recording_list.count() == 1
                and recording_list.currentItem() is not None
                and recording_list.currentItem().text() == "walk"
            ),
        )
        assert str(session.resolve()) in coordinator._watcher.directories()
        assert recording_list.count() == 1
        assert recording_list.currentItem() is not None
        assert recording_list.currentItem().text() == "walk"

        previous_count = status_spy.count()
        for cam_id in (0, 1):
            (session / f"cam_{cam_id}.mp4").touch()
        _wait_for_status_change(
            qapp,
            status_spy,
            previous_count,
            lambda: feedback_label.isHidden() and process_button.isEnabled(),
        )
        assert feedback_label.isHidden()
        assert process_button.isEnabled()

        previous_count = status_spy.count()
        (session / "cam_1.mp4").unlink()
        _wait_for_status_change(
            qapp,
            status_spy,
            previous_count,
            lambda: feedback_label.text() == "• Missing recordings/walk/cam_1.mp4." and not process_button.isEnabled(),
        )
        assert feedback_label.text() == "• Missing recordings/walk/cam_1.mp4."
        assert not process_button.isEnabled()
        assert recording_list.count() == 1
        assert recording_list.currentItem() is not None
        assert recording_list.currentItem().text() == "walk"

        previous_count = status_spy.count()
        (session / "cam_1.mp4").touch()
        _wait_for_status_change(
            qapp,
            status_spy,
            previous_count,
            lambda: feedback_label.isHidden() and process_button.isEnabled(),
        )
        assert feedback_label.isHidden()
        assert process_button.isEnabled()

        previous_count = status_spy.count()
        for cam_id in (0, 1):
            (session / f"cam_{cam_id}.mp4").unlink()
        session.rmdir()
        _wait_for_status_change(
            qapp,
            status_spy,
            previous_count,
            lambda: recording_list.count() == 0 and str(session.resolve()) not in coordinator._watcher.directories(),
        )
        assert recording_list.count() == 0
        assert str(session.resolve()) not in coordinator._watcher.directories()
        assert presenter.selected_recording is None
    finally:
        tab.cleanup()
        coordinator.cleanup()
        tracker_registry._factories.pop(tracker_name, None)
        tracker_registry._display_names.pop(tracker_name, None)
        tracker_registry._wireframes.pop(tracker_name, None)
        tracker_registry._model_cards.pop(tracker_name, None)
