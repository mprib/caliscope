"""Workspace file feedback reaching already-built views."""

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QListWidget, QPushButton, QWidget

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.gui.multi_camera_processing_tab import MultiCameraProcessingTab
from caliscope.gui.presenters.multi_camera_processing_presenter import MultiCameraProcessingState
from caliscope.gui.reconstruction_tab import ReconstructionTab
from caliscope.gui.views.camera_thumbnail_card import CameraThumbnailCard
from caliscope.gui.views.project_setup_view import ProjectSetupView
from caliscope.gui.widgets.folder_link import FolderLink
from caliscope.gui.widgets.workspace_issue_label import WorkspaceIssueLabel
from caliscope.recording.recording_validation import CameraDimensionOutcome, RecordingDimensionAssessment
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


def test_project_feedback_follows_directory_change(coordinator: WorkspaceCoordinator, qtbot) -> None:
    extrinsic = coordinator.workspace_guide.extrinsic_dir
    view = ProjectSetupView(coordinator)
    feedback = view.findChild(QLabel, "fileFeedbackLabel")
    assert feedback is not None

    assert "No extrinsic camera videos found" in feedback.text()

    (extrinsic / "cam_0.mp4").touch()
    coordinator._on_directory_changed(str(extrinsic))
    qtbot.waitUntil(lambda: 0 in coordinator.camera_array.cameras)

    assert "No extrinsic camera videos found" not in feedback.text()
    assert tuple(coordinator.camera_array.cameras) == (0,)


def test_reconstruction_folder_links_follow_selected_recording_and_results(
    coordinator: WorkspaceCoordinator,
    qapp: QApplication,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    registered_tracker: str,
) -> None:
    """Navigation links always open the current recording and its current results."""

    def matching_dimensions(_recording_dir: Path, expected_sizes):
        return RecordingDimensionAssessment(
            tuple(CameraDimensionOutcome(cam_id, size, size, None) for cam_id, size in expected_sizes)
        )

    metadata_check = MagicMock(side_effect=matching_dimensions)
    monkeypatch.setattr("caliscope.gui.views.reconstruction_widget.opengl_available", lambda: False)
    monkeypatch.setattr(
        "caliscope.gui.presenters.reconstruction_presenter.check_recording_dimensions",
        metadata_check,
    )
    coordinator.camera_array = CameraArray(
        {
            0: CameraData(cam_id=0, size=(640, 480)),
            1: CameraData(cam_id=1, size=(640, 480)),
        }
    )
    recordings = coordinator.workspace_guide.recording_dir
    for name in ("alpha", "beta"):
        session = recordings / name
        session.mkdir()
        for cam_id in (0, 1):
            (session / f"cam_{cam_id}.mp4").touch()
    coordinator._on_directory_changed(str(recordings))
    tab = ReconstructionTab(coordinator)
    tab.show()
    presenter = tab._presenter
    presenter.select_tracker(registered_tracker)
    recording_list = tab.findChild(QListWidget, "recordingList")
    assert recording_list is not None

    try:
        qtbot.waitUntil(lambda: presenter.selected_recording_is_ready)
        validated_call_count = metadata_check.call_count
        coordinator.status_changed.emit()
        qapp.processEvents()
        assert metadata_check.call_count == validated_call_count
        recording_link = next(link for link in tab.findChildren(FolderLink) if link.text() == "Open recording folder")
        results_link = next(link for link in tab.findChildren(FolderLink) if link.text() == "Open results folder")
        assert recording_link.folder == recordings / "alpha"
        assert not results_link.isVisible()

        recording_list.setCurrentRow(1)
        qtbot.waitUntil(lambda: presenter.selected_recording == "beta" and recording_link.folder == recordings / "beta")
        output_path = presenter.xyz_output_path
        assert output_path is not None
        output_path.parent.mkdir()
        output_path.touch()
        presenter.refresh_from_workspace()
        qtbot.waitUntil(lambda: results_link.isVisible() and results_link.folder == output_path.parent)
        recording_list.setCurrentRow(0)
        qtbot.waitUntil(lambda: presenter.selected_recording == "alpha" and not results_link.isVisible())
        assert results_link.folder is None
    finally:
        tab.cleanup()
        tab.close()


def test_open_extract_tab_follows_extrinsic_videos(coordinator: WorkspaceCoordinator, qtbot) -> None:
    extrinsic = coordinator.workspace_guide.extrinsic_dir
    for cam_id in (0, 1):
        (extrinsic / f"cam_{cam_id}.mp4").touch()
    coordinator.load_camera_array()
    tab = MultiCameraProcessingTab(coordinator)
    label = tab.findChild(WorkspaceIssueLabel)
    action_button = tab.findChild(QPushButton, "processingActionButton")
    assert label is not None and action_button is not None
    assert label.isHidden()
    assert _camera_card_ids(tab) == [0, 1]

    (extrinsic / "cam_1.mp4").unlink()
    coordinator._on_directory_changed(str(extrinsic))

    assert "Missing calibration/extrinsic/cam_1.mp4" in label.text()
    assert tuple(coordinator.camera_array.cameras) == (0, 1)
    assert tab._presenter is not None
    assert tab._presenter.state == MultiCameraProcessingState.UNCONFIGURED
    card = tab.findChild(CameraThumbnailCard, "cameraCard1")
    thumbnail = card.findChild(QLabel, "thumbnailLabel") if card is not None else None
    assert thumbnail is not None
    assert thumbnail.text() == "No video file for cam_1"
    assert not action_button.isEnabled()

    (extrinsic / "cam_1.mp4").touch()
    (extrinsic / "cam_2.mp4").touch()
    coordinator._on_directory_changed(str(extrinsic))
    qtbot.waitUntil(lambda: 2 in coordinator.camera_array.cameras)

    assert label.isHidden()
    assert tab._presenter.state == MultiCameraProcessingState.READY
    assert action_button.isEnabled()
    assert _camera_card_ids(tab) == [0, 1, 2]
    tab.cleanup()


def _camera_card_ids(tab: QWidget) -> list[int]:
    names = (widget.objectName() for widget in tab.findChildren(CameraThumbnailCard))
    return sorted(int(name.removeprefix("cameraCard")) for name in names)


def test_reconstruction_tab_follows_nested_recording_changes_on_disk(
    tmp_path: Path,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    registered_tracker: str,
) -> None:
    """A built tab follows nested camera files through the root watcher and session poll."""
    dimensions_match = [True]

    def matching_dimensions(_recording_dir: Path, expected_sizes):
        """Keep this structural-watcher test independent of PyAV fixture files."""
        return RecordingDimensionAssessment(
            tuple(
                CameraDimensionOutcome(
                    cam_id=cam_id,
                    expected_size=size,
                    actual_size=size if dimensions_match[0] or cam_id != 1 else (1280, 720),
                    error=None,
                )
                for cam_id, size in expected_sizes
            )
        )

    monkeypatch.setattr("caliscope.gui.views.reconstruction_widget.opengl_available", lambda: False)
    monkeypatch.setattr(
        "caliscope.gui.presenters.reconstruction_presenter.check_recording_dimensions",
        matching_dimensions,
    )
    monkeypatch.setattr("caliscope.workspace_coordinator.RECORDING_POLL_INTERVAL_MS", 50)
    monkeypatch.chdir(tmp_path.parent)
    relative_workspace = Path(tmp_path.name)
    coordinator = WorkspaceCoordinator(relative_workspace)
    coordinator.camera_array = CameraArray(
        {
            0: CameraData(cam_id=0, size=(640, 480)),
            1: CameraData(cam_id=1, size=(640, 480)),
        }
    )
    tab = ReconstructionTab(coordinator)
    presenter = tab._presenter

    recording_list = tab.findChild(QListWidget, "recordingList")
    feedback_label = tab.findChild(QLabel, "recordingFeedbackLabel")
    process_button = tab.findChild(QPushButton, "processButton")
    assert recording_list is not None
    assert feedback_label is not None
    assert process_button is not None
    presenter.select_tracker(registered_tracker)

    def selected_name() -> str | None:
        item = recording_list.currentItem()
        return item.text() if item is not None else None

    def ready_to_process() -> bool:
        return feedback_label.isHidden() and process_button.isEnabled()

    try:
        session = coordinator.workspace_guide.recording_dir / "walk"
        session.mkdir()
        qtbot.waitUntil(lambda: recording_list.count() == 1 and selected_name() == "walk")
        assert str(session.resolve()) not in coordinator._watcher.directories()

        for cam_id in (0, 1):
            (session / f"cam_{cam_id}.mp4").touch()
        qtbot.waitUntil(ready_to_process)

        renamed_session = session.with_name("stride")
        session.rename(renamed_session)
        qtbot.waitUntil(
            lambda: (
                presenter.selected_recording == "stride"
                and recording_list.count() == 1
                and selected_name() == "stride"
                and ready_to_process()
            )
        )
        session = renamed_session

        # A selected video edit produces dimension feedback and an updated
        # Process affordance. Correcting the same file recovers both.
        dimensions_match[0] = False
        os.utime(session / "cam_1.mp4", None)
        qtbot.waitUntil(lambda: "1280×720" in feedback_label.text() and not process_button.isEnabled())
        dimensions_match[0] = True
        os.utime(session / "cam_1.mp4", None)
        qtbot.waitUntil(ready_to_process)

        (session / "cam_1.mp4").unlink()
        qtbot.waitUntil(
            lambda: feedback_label.text() == "Missing recordings/stride/cam_1.mp4." and not process_button.isEnabled()
        )
        assert recording_list.count() == 1
        assert selected_name() == "stride"

        (session / "cam_1.mp4").touch()
        qtbot.waitUntil(ready_to_process)

        for cam_id in (0, 1):
            (session / f"cam_{cam_id}.mp4").unlink()
        session.rmdir()
        qtbot.waitUntil(lambda: recording_list.count() == 0)
        assert presenter.selected_recording is None

        session.mkdir()
        qtbot.waitUntil(lambda: recording_list.count() == 1)
    finally:
        tab.cleanup()
        coordinator.cleanup()
