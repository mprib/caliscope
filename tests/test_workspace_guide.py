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


def _touch_ports(directory: Path, ports: list[int]) -> None:
    """Create empty port_N.mp4 files for each port number."""
    for p in ports:
        (directory / f"port_{p}.mp4").touch()


class TestMissingFilesInDir:
    """Verify missing_files_in_dir compares against actual port sets, not 1-based ranges."""

    def test_zero_indexed_ports_all_present(self, workspace: Path) -> None:
        """Ports 0-3 in both dirs should report NONE missing — the original bug."""
        _touch_ports(workspace / "calibration" / "extrinsic", [0, 1, 2, 3])
        _touch_ports(workspace / "calibration" / "intrinsic", [0, 1, 2, 3])

        guide = WorkspaceGuide(workspace)
        assert guide.all_instrinsic_mp4s_available() is True

    def test_one_indexed_ports_all_present(self, workspace: Path) -> None:
        _touch_ports(workspace / "calibration" / "extrinsic", [1, 2, 3, 4])
        _touch_ports(workspace / "calibration" / "intrinsic", [1, 2, 3, 4])

        guide = WorkspaceGuide(workspace)
        assert guide.all_instrinsic_mp4s_available() is True

    def test_missing_intrinsic_port(self, workspace: Path) -> None:
        _touch_ports(workspace / "calibration" / "extrinsic", [0, 1, 2, 3])
        _touch_ports(workspace / "calibration" / "intrinsic", [0, 1, 3])  # missing port 2

        guide = WorkspaceGuide(workspace)
        assert guide.all_instrinsic_mp4s_available() is False
        assert (
            guide.missing_files_in_dir(workspace / "calibration" / "intrinsic", guide.get_camera_ports())
            == "port_2.mp4"
        )

    def test_noncontiguous_ports(self, workspace: Path) -> None:
        """Ports with gaps (e.g. 0, 2, 5) should work correctly."""
        _touch_ports(workspace / "calibration" / "extrinsic", [0, 2, 5])
        _touch_ports(workspace / "calibration" / "intrinsic", [0, 2, 5])

        guide = WorkspaceGuide(workspace)
        assert guide.all_instrinsic_mp4s_available() is True

    def test_empty_extrinsic_dir(self, workspace: Path) -> None:
        guide = WorkspaceGuide(workspace)
        assert guide.all_instrinsic_mp4s_available() is False
        assert guide.all_extrinsic_mp4s_available() is False

    def test_directory_does_not_exist(self, tmp_path: Path) -> None:
        guide = WorkspaceGuide(tmp_path / "nonexistent")
        assert guide.all_instrinsic_mp4s_available() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
