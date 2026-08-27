"""Tests for WorkspaceGuide filesystem inspection."""

from pathlib import Path

import pytest

from caliscope.workspace_guide import WorkspaceGuide


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a workspace directory with extrinsic and intrinsic subdirectories."""
    extrinsic = tmp_path / "calibration" / "extrinsic"
    intrinsic = tmp_path / "calibration" / "intrinsic"
    extrinsic.mkdir(parents=True)
    intrinsic.mkdir(parents=True)
    return tmp_path


def _touch_cam_ids(directory: Path, cam_ids: list[int]) -> None:
    """Create empty cam_N.mp4 files for each cam_id number."""
    for cam_id in cam_ids:
        (directory / f"cam_{cam_id}.mp4").touch()


class TestMissingFilesInDir:
    """Verify missing_files_in_dir compares against actual camera sets, not 1-based ranges."""

    def test_zero_indexed_ports_all_present(self, workspace: Path) -> None:
        """Ports 0-3 in both dirs should report NONE missing — the original bug."""
        _touch_cam_ids(workspace / "calibration" / "extrinsic", [0, 1, 2, 3])
        _touch_cam_ids(workspace / "calibration" / "intrinsic", [0, 1, 2, 3])

        guide = WorkspaceGuide(workspace)
        assert guide.all_instrinsic_mp4s_available() is True

    def test_one_indexed_ports_all_present(self, workspace: Path) -> None:
        _touch_cam_ids(workspace / "calibration" / "extrinsic", [1, 2, 3, 4])
        _touch_cam_ids(workspace / "calibration" / "intrinsic", [1, 2, 3, 4])

        guide = WorkspaceGuide(workspace)
        assert guide.all_instrinsic_mp4s_available() is True

    def test_missing_intrinsic_port(self, workspace: Path) -> None:
        _touch_cam_ids(workspace / "calibration" / "extrinsic", [0, 1, 2, 3])
        _touch_cam_ids(workspace / "calibration" / "intrinsic", [0, 1, 3])  # missing cam 2

        guide = WorkspaceGuide(workspace)
        assert guide.all_instrinsic_mp4s_available() is False
        assert guide.missing_files_in_dir(workspace / "calibration" / "intrinsic", guide.get_cam_ids()) == "cam_2.mp4"

    def test_noncontiguous_ports(self, workspace: Path) -> None:
        """Ports with gaps (e.g. 0, 2, 5) should work correctly."""
        _touch_cam_ids(workspace / "calibration" / "extrinsic", [0, 2, 5])
        _touch_cam_ids(workspace / "calibration" / "intrinsic", [0, 2, 5])

        guide = WorkspaceGuide(workspace)
        assert guide.all_instrinsic_mp4s_available() is True

    def test_empty_extrinsic_dir(self, workspace: Path) -> None:
        guide = WorkspaceGuide(workspace)
        assert guide.all_instrinsic_mp4s_available() is False
        assert guide.all_extrinsic_mp4s_available() is False

    def test_directory_does_not_exist(self, tmp_path: Path) -> None:
        guide = WorkspaceGuide(tmp_path / "nonexistent")
        assert guide.all_instrinsic_mp4s_available() is False


class TestCameraVideoAssessment:
    def test_each_issue_kind_is_reported_with_a_relative_path(self, workspace: Path) -> None:
        extrinsic = workspace / "calibration" / "extrinsic"
        _touch_cam_ids(extrinsic, [0, 3])
        (extrinsic / "cam_1.MP4").touch()
        (extrinsic / "calibration_take.mp4").touch()
        (extrinsic / "timestamps.csv").touch()

        assessment = WorkspaceGuide(workspace).assess_extrinsic_videos([0, 5])

        assert assessment.camera_ids == (0, 3)
        assert assessment.missing_camera_ids == (5,)
        assert [(issue.code, issue.relative_path) for issue in assessment.issues] == [
            ("missing_camera_video", "calibration/extrinsic/cam_5.mp4"),
            ("unexpected_mp4", "calibration/extrinsic/cam_3.mp4"),
            ("unexpected_mp4", "calibration/extrinsic/calibration_take.mp4"),
            ("malformed_camera_filename", "calibration/extrinsic/cam_1.MP4"),
        ]


class TestRecordingLayoutIssues:
    def test_misplaced_layouts_get_hints(self, workspace: Path) -> None:
        recordings = workspace / "recordings"
        for cam_id in (0, 1):
            camera_dir = recordings / f"cam_{cam_id}"
            camera_dir.mkdir(parents=True)
            (camera_dir / "take.mp4").touch()
        (recordings / "cam_0.mp4").touch()
        (recordings / "timestamps.csv").touch()

        issues = WorkspaceGuide(workspace).recording_layout_issues()

        assert [issue.code for issue in issues] == [
            "recordings_need_session_folder",
            "recording_timestamps_need_session_folder",
            "recording_split_by_camera",
        ]

    def test_single_camera_split_directory_is_recognized_from_contents(self, workspace: Path) -> None:
        camera_dir = workspace / "recordings" / "cam_7"
        camera_dir.mkdir(parents=True)
        (camera_dir / "walk_trial.mp4").touch()

        issues = WorkspaceGuide(workspace).recording_layout_issues()

        assert [issue.code for issue in issues] == ["recording_split_by_camera"]

    def test_canonical_camera_partition_directories_are_recognized(self, workspace: Path) -> None:
        recordings = workspace / "recordings"
        for cam_id in (0, 1):
            camera_dir = recordings / f"cam_{cam_id}"
            camera_dir.mkdir(parents=True)
            (camera_dir / f"cam_{cam_id}.mp4").touch()

        issues = WorkspaceGuide(workspace).recording_layout_issues()

        assert [issue.code for issue in issues] == ["recording_split_by_camera"]

    def test_session_name_is_not_constrained(self, workspace: Path) -> None:
        session = workspace / "recordings" / "cam_7"
        session.mkdir(parents=True)
        _touch_cam_ids(session, [0, 7])

        guide = WorkspaceGuide(workspace)

        assert guide.recording_layout_issues() == ()
        assert guide.assess_recordings([0, 7])["cam_7"].is_ready is True


class TestRecordingAssessments:
    def test_all_immediate_sessions_are_returned_and_timestamps_are_optional(self, workspace: Path) -> None:
        recordings = workspace / "recordings"
        ready = recordings / "ready"
        without_timestamps = recordings / "without_timestamps"
        missing_camera = recordings / "missing_camera"
        empty = recordings / "empty"
        for session in (ready, without_timestamps, missing_camera, empty):
            session.mkdir(parents=True)

        _touch_cam_ids(ready, [0, 2])
        (ready / "timestamps.csv").touch()
        _touch_cam_ids(without_timestamps, [0, 2])
        _touch_cam_ids(missing_camera, [0])

        guide = WorkspaceGuide(workspace)
        assessments = guide.assess_recordings([0, 2])

        assert list(assessments) == ["empty", "missing_camera", "ready", "without_timestamps"]
        assert assessments["ready"].is_ready is True
        assert assessments["without_timestamps"].is_ready is True
        assert assessments["missing_camera"].missing_camera_ids == (2,)
        assert assessments["empty"].missing_camera_ids == (0, 2)
        assert guide.ready_recording_dirs([0, 2]) == ["ready", "without_timestamps"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
