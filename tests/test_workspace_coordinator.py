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
from PySide6.QtCore import QElapsedTimer, QEventLoop
from PySide6.QtTest import QSignalSpy

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.workspace_coordinator import WorkspaceCoordinator


def _wait_for(qapp, condition, timeout_ms: int = 3000) -> None:
    """Process Qt filesystem events until a bounded condition becomes true."""
    timer = QElapsedTimer()
    timer.start()
    while not condition():
        qapp.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if timer.elapsed() >= timeout_ms:
            raise AssertionError("Timed out waiting for filesystem watcher event")


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


def test_reconstruction_status_counts_all_sessions_and_ready_sessions(
    coordinator: WorkspaceCoordinator,
):
    coordinator.camera_array = CameraArray(
        {
            0: CameraData(cam_id=0, size=(640, 480)),
            2: CameraData(cam_id=2, size=(640, 480)),
        }
    )
    recordings = coordinator.workspace_guide.recording_dir
    ready = recordings / "ready"
    partial = recordings / "partial"
    empty = recordings / "empty"
    for session in (ready, partial, empty):
        session.mkdir()
    for cam_id in (0, 2):
        (ready / f"cam_{cam_id}.mp4").touch()
    (partial / "cam_0.mp4").touch()

    status = coordinator.get_workflow_status()

    assert status.recording_names == ["empty", "partial", "ready"]
    assert status.ready_recording_names == ["ready"]
    assert status.recordings_available is True
    assert [issue.relative_path for issue in status.recording_issues] == [
        "recordings/empty/cam_0.mp4",
        "recordings/empty/cam_2.mp4",
        "recordings/partial/cam_2.mp4",
    ]


def test_reconstruction_tab_only_requires_capture_volume_bundle(
    coordinator: WorkspaceCoordinator,
):
    assert coordinator.workspace_guide.assess_recordings([]) == {}
    coordinator.capture_volume_repository.camera_array_path.touch()

    assert coordinator.reconstruction_tab_enabled is True


def test_recording_session_watches_follow_root_directory_changes(
    coordinator: WorkspaceCoordinator,
):
    """Session watches are the coordinator's record, reconciled on every root change.

    Qt's own directories() list is not asserted here: fsevents and the Windows
    engine only drop a deleted directory asynchronously, through the event loop.
    """
    recording_dir = str(coordinator.workspace_guide.recording_dir)
    session = coordinator.workspace_guide.recording_dir / "walk"
    session.mkdir()

    coordinator._on_directory_changed(recording_dir)

    assert str(session.resolve()) in coordinator._session_watches

    session.rmdir()
    coordinator._on_directory_changed(recording_dir)

    assert str(session.resolve()) not in coordinator._session_watches


@pytest.mark.parametrize("linked_inputs", [False, True], ids=["regular", "linked-session-and-video"])
def test_atomic_recording_video_replacement_readds_file_watch(
    coordinator: WorkspaceCoordinator,
    qapp,
    tmp_path: Path,
    linked_inputs: bool,
):
    """Reconciliation restores a dropped Qt file watch after atomic replacement."""
    session = coordinator.workspace_guide.recording_dir / "walk"
    if linked_inputs:
        session_target = tmp_path / "external-session"
        video_target = tmp_path / "external-cam_4.mp4"
        session_target.mkdir()
        video_target.write_bytes(b"first")
        try:
            session.symlink_to(session_target, target_is_directory=True)
            video = session / "cam_4.mp4"
            video.symlink_to(video_target)
        except OSError as error:
            pytest.skip(f"symlinks are unavailable: {error}")
    else:
        session.mkdir()
        video = session / "cam_4.mp4"
        video.write_bytes(b"first")
    coordinator._on_directory_changed(str(coordinator.workspace_guide.recording_dir))
    assert str(video.resolve()) in coordinator._watcher.files()

    directory_spy = QSignalSpy(coordinator.recording_directory_changed)

    def session_changed_since(previous_count: int) -> bool:
        return any(
            Path(directory_spy.at(index)[0]) == session.absolute()
            for index in range(previous_count, directory_spy.count())
        )

    (session / "notes.txt").touch()
    _wait_for(qapp, lambda: session_changed_since(0))

    video_spy = QSignalSpy(coordinator.recording_video_changed)
    previous_count = video_spy.count()
    video.write_bytes(b"changed")
    _wait_for(qapp, lambda: video_spy.count() > previous_count)
    assert Path(video_spy.at(video_spy.count() - 1)[0]) == video.resolve()

    previous_directory_count = directory_spy.count()
    replacement = session / "replacement.tmp"
    replacement.write_bytes(b"replacement")
    replacement.replace(video)

    _wait_for(qapp, lambda: session_changed_since(previous_directory_count))

    previous_video_count = video_spy.count()
    video.write_bytes(b"replacement changed")
    _wait_for(qapp, lambda: video_spy.count() > previous_video_count)
    assert Path(video_spy.at(video_spy.count() - 1)[0]) == video.resolve()


def test_rotation_from_multi_camera_presenter_is_persisted(tmp_path: Path, qapp):
    """The presenter must not mutate the coordinator's cameras in place, or the
    coordinator sees no change and skips the save."""
    camera = CameraData(cam_id=0, size=(640, 480), matrix=np.eye(3), distortions=np.zeros(5))
    coordinator = WorkspaceCoordinator(tmp_path)
    coordinator.camera_repository.save(CameraArray({0: camera}))
    coordinator.load_camera_array()
    assert coordinator.camera_array.cameras[0].rotation_count == 0

    presenter = coordinator.create_multi_camera_presenter()
    presenter.rotation_changed.connect(coordinator.persist_camera_rotation)
    presenter.set_cameras(coordinator.camera_array.cameras)
    calibration_spy = QSignalSpy(coordinator.calibration_changed)

    presenter.set_rotation(0, 1)

    assert coordinator.camera_repository.load().cameras[0].rotation_count == 1
    assert calibration_spy.count() == 1
    presenter.cleanup()
