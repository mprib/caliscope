"""Tests for video_utils.read_video_properties (PyAV backend)."""

from pathlib import Path

import pytest

from caliscope.recording.video_utils import read_video_properties

EXTRINSIC_DIR = Path(__file__).parent / "sessions" / "4_cam_recording" / "calibration" / "extrinsic"


class TestReadVideoProperties:
    """read_video_properties returns correct metadata and raises on bad input."""

    def test_returns_correct_metadata_for_test_video(self):
        """Reads a real test video and validates all fields are sensible."""
        video_path = EXTRINSIC_DIR / "cam_0.mp4"
        props = read_video_properties(video_path)

        assert props["fps"] > 0
        assert props["frame_count"] > 0
        assert props["width"] > 0
        assert props["height"] > 0
        assert props["size"] == (props["width"], props["height"])

    def test_file_not_found_raises(self, tmp_path: Path):
        """FileNotFoundError for nonexistent path."""
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            read_video_properties(tmp_path / "nonexistent.mp4")

    def test_corrupt_file_raises_value_error(self, tmp_path: Path):
        """ValueError (not av.error.*) for a file that isn't valid video."""
        fake_video = tmp_path / "not_a_video.mp4"
        fake_video.write_text("this is not a video file")

        with pytest.raises(ValueError, match="Could not open video file"):
            read_video_properties(fake_video)

    def test_consistent_across_cameras(self):
        """All 4 test cameras return plausible, consistent FPS."""
        fps_values = []
        for cam_id in range(4):
            props = read_video_properties(EXTRINSIC_DIR / f"cam_{cam_id}.mp4")
            fps_values.append(props["fps"])

        # All cameras should have the same FPS (within rounding)
        for fps in fps_values:
            assert abs(fps - fps_values[0]) < 0.1


if __name__ == "__main__":
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for cam_id in range(4):
        path = EXTRINSIC_DIR / f"cam_{cam_id}.mp4"
        props = read_video_properties(path)
        print(f"cam_{cam_id}: {props}")
