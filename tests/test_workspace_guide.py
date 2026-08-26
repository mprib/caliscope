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
    def test_canonical_noncontiguous_camera_ids_are_accepted(self, workspace: Path) -> None:
        _touch_cam_ids(workspace / "calibration" / "extrinsic", [0, 3, 12])

        assessment = WorkspaceGuide(workspace).assess_extrinsic_videos(())

        assert assessment.camera_ids == (0, 3, 12)
        assert assessment.issues == ()

    @pytest.mark.parametrize(
        "filename",
        ["cam_01.mp4", "cam_-1.mp4", "cam_1_extra.mp4", "cam1.mp4", "cam_1.MP4"],
    )
    def test_near_match_is_reported_as_malformed(self, workspace: Path, filename: str) -> None:
        extrinsic = workspace / "calibration" / "extrinsic"
        (extrinsic / "cam_0.mp4").touch()
        (extrinsic / filename).touch()

        assessment = WorkspaceGuide(workspace).assess_extrinsic_videos(())

        assert assessment.camera_ids == (0,)
        issue = next(issue for issue in assessment.issues if issue.code == "malformed_camera_filename")
        assert issue.relative_path == f"calibration/extrinsic/{filename}"
        assert "must be named cam_N.mp4" in issue.message

    def test_unexpected_mp4_is_reported(self, workspace: Path) -> None:
        extrinsic = workspace / "calibration" / "extrinsic"
        (extrinsic / "cam_0.mp4").touch()
        (extrinsic / "calibration_take.mp4").touch()
        (extrinsic / "timestamps.csv").touch()

        assessment = WorkspaceGuide(workspace).assess_extrinsic_videos(())

        assert [issue.code for issue in assessment.issues] == ["unexpected_mp4"]
        assert assessment.issues[0].relative_path == "calibration/extrinsic/calibration_take.mp4"

    def test_missing_intrinsic_video_uses_relative_path(self, workspace: Path) -> None:
        _touch_cam_ids(workspace / "calibration" / "extrinsic", [0, 5])
        _touch_cam_ids(workspace / "calibration" / "intrinsic", [0])
        guide = WorkspaceGuide(workspace)

        assessment = guide.assess_intrinsic_videos(guide.get_cam_ids())

        assert assessment.missing_camera_ids == (5,)
        assert assessment.issues[0].code == "missing_camera_video"
        assert assessment.issues[0].relative_path == "calibration/intrinsic/cam_5.mp4"

    def test_missing_extrinsic_video_is_reported_against_expected_set(self, workspace: Path) -> None:
        _touch_cam_ids(workspace / "calibration" / "extrinsic", [0])

        assessment = WorkspaceGuide(workspace).assess_extrinsic_videos([0, 1])

        assert assessment.camera_ids == (0,)
        assert assessment.missing_camera_ids == (1,)
        assert [issue.code for issue in assessment.issues] == ["missing_camera_video"]
        assert assessment.issues[0].relative_path == "calibration/extrinsic/cam_1.mp4"

    def test_empty_extrinsic_directory_is_not_ready(self, workspace: Path) -> None:
        assessment = WorkspaceGuide(workspace).assess_extrinsic_videos(())

        assert assessment.camera_ids == ()
        assert assessment.issues[0].code == "missing_camera_video"
        assert assessment.issues[0].relative_path == "calibration/extrinsic"

    def test_canonical_name_must_be_a_direct_child_file(self, workspace: Path) -> None:
        nested = workspace / "calibration" / "extrinsic" / "cam_0.mp4"
        nested.mkdir()

        assessment = WorkspaceGuide(workspace).assess_extrinsic_videos(())

        assert assessment.camera_ids == ()


class TestRecordingLayoutIssues:
    def test_root_level_recording_videos_get_session_folder_hint(self, workspace: Path) -> None:
        recordings = workspace / "recordings"
        recordings.mkdir()
        (recordings / "cam_0.mp4").touch()

        issues = WorkspaceGuide(workspace).recording_layout_issues()

        assert [issue.code for issue in issues] == ["recordings_need_session_folder"]

    def test_camera_folders_get_combined_session_hint(self, workspace: Path) -> None:
        recordings = workspace / "recordings"
        for cam_id in (0, 1):
            camera_dir = recordings / f"cam_{cam_id}"
            camera_dir.mkdir(parents=True)
            (camera_dir / "take.mp4").touch()

        issues = WorkspaceGuide(workspace).recording_layout_issues()

        assert [issue.code for issue in issues] == ["recording_split_by_camera"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
