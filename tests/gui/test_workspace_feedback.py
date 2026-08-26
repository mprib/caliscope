"""Focused Project view tests for workspace file feedback."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QObject, Signal

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.workflow_status import WorkspaceIssue
from caliscope.gui.cameras_tab_widget import CamerasTabWidget
from caliscope.gui.multi_camera_processing_tab import MultiCameraProcessingTab
from caliscope.gui.presenters.multi_camera_processing_presenter import MultiCameraProcessingState
from caliscope.gui.views.project_setup_view import ProjectSetupView
from caliscope.gui.widgets.cameras_info_placeholder import CamerasInfoPlaceholder
from caliscope.workspace_coordinator import WorkspaceCoordinator


class _CameraTabCoordinator(QObject):
    status_changed = Signal()
    intrinsic_target_changed = Signal()

    def __init__(self, camera_array: CameraArray) -> None:
        super().__init__()
        self.camera_array = camera_array
        self.intrinsic_frame_skip = 5
        self.intrinsic_cam_ids = set(camera_array.cameras)
        self.intrinsic_video_issues: tuple[WorkspaceIssue, ...] = ()
        self.workspace_guide = SimpleNamespace(
            intrinsic_dir=Path("calibration/intrinsic"),
            get_cam_ids_in_dir=lambda _directory: sorted(self.intrinsic_cam_ids),
        )

    def get_workflow_status(self):
        return SimpleNamespace(intrinsic_video_issues=self.intrinsic_video_issues)


def test_project_feedback_follows_directory_change(tmp_path: Path, qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        "caliscope.workspace_coordinator.read_video_properties",
        lambda _path: {"size": (640, 480)},
    )
    coordinator = WorkspaceCoordinator(tmp_path)
    extrinsic = coordinator.workspace_guide.extrinsic_dir
    view = ProjectSetupView(coordinator)

    assert "No extrinsic camera videos found" in view._file_feedback_label.text()

    (extrinsic / "cam_0.mp4").touch()
    coordinator._on_directory_changed(str(extrinsic))

    assert view._file_feedback_label.text() == (
        "Calibration video file names and locations look correct.\nIntrinsic videos: none found."
    )
    assert tuple(coordinator.camera_array.cameras) == (0,)
    assert view._intrinsic_row._status_label.text().startswith("No intrinsic video files in workspace.")

    (coordinator.workspace_guide.intrinsic_dir / "cam_0.mp4").touch()
    coordinator._on_directory_changed(str(coordinator.workspace_guide.intrinsic_dir))

    assert view._intrinsic_row._status_label.text() == "Video files in workspace for cameras 0; 0/1 calibrated"
    coordinator.cleanup()


def test_project_names_precalibrated_intrinsics_without_videos(tmp_path: Path, qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        "caliscope.workspace_coordinator.read_video_properties",
        lambda _path: {"size": (640, 480)},
    )
    coordinator = WorkspaceCoordinator(tmp_path)
    extrinsic = coordinator.workspace_guide.extrinsic_dir
    cameras = {
        cam_id: CameraData(cam_id=cam_id, size=(640, 480), matrix=np.eye(3), distortions=np.zeros(5))
        for cam_id in (0, 1)
    }
    coordinator.camera_repository.save(CameraArray(cameras))
    for cam_id in (0, 1):
        (extrinsic / f"cam_{cam_id}.mp4").touch()
    coordinator.load_camera_array()
    view = ProjectSetupView(coordinator)

    assert view._intrinsic_row._status_label.text() == (
        "Intrinsics loaded from camera_array.toml for cameras 0, 1; no intrinsic video files in workspace"
    )

    placeholder = CamerasInfoPlaceholder(coordinator)
    text = placeholder._label.text()
    assert "Intrinsics loaded from camera_array.toml" in text
    assert "Cam 0" in text and "Cam 1" in text
    assert "experimental" not in text

    coordinator.camera_array.cameras[1].matrix = None
    coordinator.status_changed.emit()
    text = placeholder._label.text()
    assert "Cameras 1 have no intrinsics" in text
    assert "experimental" in text
    coordinator.cleanup()


def test_placeholder_without_intrinsics_states_the_risk(tmp_path: Path, qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        "caliscope.workspace_coordinator.read_video_properties",
        lambda _path: {"size": (640, 480)},
    )
    coordinator = WorkspaceCoordinator(tmp_path)
    (coordinator.workspace_guide.extrinsic_dir / "cam_0.mp4").touch()
    coordinator.load_camera_array()

    placeholder = CamerasInfoPlaceholder(coordinator)
    text = placeholder._label.text()

    assert "No intrinsic calibration videos" in text
    assert "experimental" in text
    assert "camera_array.toml" not in text
    coordinator.cleanup()


def test_open_extract_tab_warns_when_extrinsic_video_disappears(tmp_path: Path, qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        "caliscope.workspace_coordinator.read_video_properties",
        lambda _path: {"size": (640, 480)},
    )
    coordinator = WorkspaceCoordinator(tmp_path)
    extrinsic = coordinator.workspace_guide.extrinsic_dir
    for cam_id in (0, 1):
        (extrinsic / f"cam_{cam_id}.mp4").touch()
    coordinator.load_camera_array()
    tab = MultiCameraProcessingTab(coordinator)

    assert tab._file_warning_label.isHidden()

    (extrinsic / "cam_1.mp4").unlink()
    coordinator._on_directory_changed(str(coordinator.workspace_guide.extrinsic_dir))

    assert not tab._file_warning_label.isHidden()
    assert "Missing calibration/extrinsic/cam_1.mp4" in tab._file_warning_label.text()
    assert tuple(coordinator.camera_array.cameras) == (0, 1)
    assert tab._presenter is not None and tab._widget is not None
    assert tab._presenter.state == MultiCameraProcessingState.UNCONFIGURED
    assert tab._widget._camera_cards[1]._thumbnail_label.text() == "No video file for cam_1"
    assert not tab._widget._action_btn.isEnabled()

    (extrinsic / "cam_1.mp4").touch()
    coordinator._on_directory_changed(str(coordinator.workspace_guide.extrinsic_dir))

    assert tab._file_warning_label.isHidden()
    assert tab._presenter.state == MultiCameraProcessingState.READY
    assert tab._widget._action_btn.isEnabled()
    tab.cleanup()
    coordinator.cleanup()


def test_open_extract_tab_adds_card_when_camera_appears(tmp_path: Path, qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        "caliscope.workspace_coordinator.read_video_properties",
        lambda _path: {"size": (640, 480)},
    )
    coordinator = WorkspaceCoordinator(tmp_path)
    extrinsic = coordinator.workspace_guide.extrinsic_dir
    (extrinsic / "cam_0.mp4").touch()
    coordinator.load_camera_array()
    tab = MultiCameraProcessingTab(coordinator)
    assert tab._widget is not None
    assert sorted(tab._widget._camera_cards) == [0]

    (extrinsic / "cam_2.mp4").touch()
    coordinator._on_directory_changed(str(extrinsic))

    assert sorted(tab._widget._camera_cards) == [0, 2]
    tab.cleanup()
    coordinator.cleanup()


def test_open_intrinsics_tab_refreshes_when_camera_is_added(qapp, monkeypatch) -> None:
    coordinator = _CameraTabCoordinator(CameraArray({0: CameraData(cam_id=0, size=(640, 480))}))
    monkeypatch.setattr(CamerasTabWidget, "_update_pattern_preview", lambda _self: None)
    monkeypatch.setattr(CamerasTabWidget, "_on_camera_selected", lambda _self, _cam_id: None)
    tab = CamerasTabWidget(coordinator)  # type: ignore[arg-type]

    assert tab.camera_list.count() == 1

    coordinator.camera_array.cameras[2] = CameraData(cam_id=2, size=(640, 480))
    coordinator.intrinsic_cam_ids.add(2)
    coordinator.status_changed.emit()

    assert tab.camera_list.count() == 2
    assert tab.camera_list.item(1).text() == "○ Cam 2"
    tab.close()


def test_open_intrinsics_tab_removes_camera_when_video_disappears(qapp, monkeypatch) -> None:
    coordinator = _CameraTabCoordinator(
        CameraArray(
            {
                0: CameraData(cam_id=0, size=(640, 480)),
                2: CameraData(cam_id=2, size=(640, 480)),
            }
        )
    )
    monkeypatch.setattr(CamerasTabWidget, "_update_pattern_preview", lambda _self: None)
    monkeypatch.setattr(CamerasTabWidget, "_on_camera_selected", lambda _self, _cam_id: None)
    tab = CamerasTabWidget(coordinator)  # type: ignore[arg-type]

    assert tab.camera_list.count() == 2

    coordinator.intrinsic_cam_ids.remove(2)
    coordinator.intrinsic_video_issues = (
        WorkspaceIssue(
            code="missing_camera_video",
            message="Missing calibration/intrinsic/cam_2.mp4.",
            relative_path="calibration/intrinsic/cam_2.mp4",
        ),
    )
    coordinator.status_changed.emit()

    assert tab.camera_list.count() == 1
    assert tab.camera_list.item(0).text() == "○ Cam 0"
    assert "Missing calibration/intrinsic/cam_2.mp4" in tab._file_warning_label.text()
    assert not tab._file_warning_label.isHidden()
    tab.close()
