"""Tests for WorkspaceCoordinator's extrinsic constraint-factory wiring.

Charuco calibration used to run with constraints=None: board geometry seeded
PnP but never entered bundle adjustment as a constraint. These tests confirm
create_extrinsic_calibration_presenter() wires a constraint factory that
compiles board-geometry distance constraints for the default (charuco)
target, and still wires the ArUco marker-set factory for the ArUco target.
"""

from pathlib import Path

import numpy as np
import pytest

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.workspace_coordinator import WorkspaceCoordinator


@pytest.fixture
def coordinator(tmp_path: Path, qapp) -> WorkspaceCoordinator:
    return WorkspaceCoordinator(tmp_path)


def test_charuco_extrinsic_presenter_gets_board_geometry_constraints(coordinator: WorkspaceCoordinator):
    """Default routing is charuco/charuco; the presenter's constraint factory
    should compile a non-empty ConstraintSet from board geometry, not None."""
    assert coordinator.targets_repository.extrinsic_target_type == "charuco"

    presenter = coordinator.create_extrinsic_calibration_presenter()

    assert presenter._constraint_factory is not None
    constraints = presenter._constraint_factory()
    assert constraints is not None
    assert len(constraints.distances) > 0
    assert constraints.static_object_ids == frozenset()
    assert constraints.centroid_distances == ()
    assert presenter._extrinsic_target_type == "charuco"


def test_aruco_extrinsic_presenter_gets_marker_set_constraints(coordinator: WorkspaceCoordinator):
    """Switching routing to aruco keeps the marker-set constraint factory
    (regression check for the branch this task modified)."""
    routing = coordinator.targets_repository.get_routing()
    coordinator.targets_repository.save_routing(
        type(routing)(
            intrinsic_target_type=routing.intrinsic_target_type,
            extrinsic_target_type="aruco",
            extrinsic_charuco_same_as_intrinsic=routing.extrinsic_charuco_same_as_intrinsic,
        )
    )

    presenter = coordinator.create_extrinsic_calibration_presenter()

    assert presenter._constraint_factory is not None
    constraints = presenter._constraint_factory()
    assert constraints is not None
    assert len(constraints.distances) > 0
    assert presenter._extrinsic_target_type == "aruco"


def test_empty_workspace_video_status_is_not_ready(coordinator: WorkspaceCoordinator):
    status = coordinator.get_workflow_status()

    assert status.intrinsic_videos_available is False
    assert status.extrinsic_videos_available is False
    assert status.extrinsic_video_issues[0].code == "missing_camera_video"


def test_deleted_extrinsic_video_is_reported_as_missing(
    coordinator: WorkspaceCoordinator,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "caliscope.workspace_coordinator.read_video_properties",
        lambda _path: {"size": (640, 480)},
    )
    extrinsic = coordinator.workspace_guide.extrinsic_dir
    intrinsic = coordinator.workspace_guide.intrinsic_dir
    for cam_id in (0, 1):
        (extrinsic / f"cam_{cam_id}.mp4").touch()
        (intrinsic / f"cam_{cam_id}.mp4").touch()
    coordinator.load_camera_array()

    (extrinsic / "cam_1.mp4").unlink()
    status = coordinator.get_workflow_status()

    assert status.camera_count == 2
    assert status.extrinsic_videos_available is False
    assert status.extrinsic_videos_missing == [1]
    assert [issue.relative_path for issue in status.extrinsic_video_issues] == ["calibration/extrinsic/cam_1.mp4"]
    assert status.intrinsic_videos_missing == []
    assert status.intrinsic_video_issues == ()


def test_directory_change_discovers_cameras_and_keeps_persisted_calibration(
    coordinator: WorkspaceCoordinator,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "caliscope.workspace_coordinator.read_video_properties",
        lambda _path: {"size": (640, 480)},
    )
    calibrated = CameraData(cam_id=0, size=(640, 480), matrix=np.eye(3), distortions=np.zeros(5))
    coordinator.camera_repository.save(CameraArray({0: calibrated}))
    extrinsic = coordinator.workspace_guide.extrinsic_dir
    intrinsic = coordinator.workspace_guide.intrinsic_dir
    for cam_id in (0, 1):
        (extrinsic / f"cam_{cam_id}.mp4").touch()
        (intrinsic / f"cam_{cam_id}.mp4").touch()

    assert coordinator.camera_array.cameras == {}
    assert coordinator.multi_camera_tab_enabled is False

    coordinator._on_directory_changed(str(extrinsic))

    assert tuple(coordinator.camera_array.cameras) == (0, 1)
    assert coordinator.camera_array.cameras[0].matrix is not None
    assert coordinator.camera_repository.load().cameras[0].matrix is not None
    assert coordinator.cameras_tab_enabled is True
    assert coordinator.multi_camera_tab_enabled is True
