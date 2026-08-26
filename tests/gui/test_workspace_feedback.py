"""Workspace file feedback reaching already-built views."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from caliscope.gui.multi_camera_processing_tab import MultiCameraProcessingTab
from caliscope.gui.presenters.multi_camera_processing_presenter import MultiCameraProcessingState
from caliscope.gui.views.project_setup_view import ProjectSetupView
from caliscope.workspace_coordinator import WorkspaceCoordinator


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
