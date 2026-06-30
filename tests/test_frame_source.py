"""Tests for FrameSource - forward-only video frame access."""

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from caliscope.recording.frame_source import FrameSource


# Test video directory and camera
TEST_VIDEO_DIR = Path(__file__).parent / "sessions/4_cam_recording/calibration/extrinsic"
TEST_CAM_ID = 0


@pytest.fixture
def frame_source() -> Generator[FrameSource, None, None]:
    """Create a FrameSource for testing."""
    source = FrameSource(TEST_VIDEO_DIR, TEST_CAM_ID)
    yield source
    source.close()


class TestFrameSourceProperties:
    """Test FrameSource metadata properties."""

    def test_opens_video_file(self, frame_source: FrameSource) -> None:
        """FrameSource opens video and exposes basic metadata."""
        assert frame_source.frame_count > 0
        assert frame_source.fps > 0
        assert frame_source.size[0] > 0  # width
        assert frame_source.size[1] > 0  # height

    def test_start_frame_index_is_zero(self, frame_source: FrameSource) -> None:
        """start_frame_index is always 0 for raw video."""
        assert frame_source.start_frame_index == 0

    def test_last_frame_index_tracks_frame_count(self, frame_source: FrameSource) -> None:
        """last_frame_index is the metadata-derived final index."""
        assert frame_source.last_frame_index == frame_source.frame_count - 1

    def test_cam_id_stored(self, frame_source: FrameSource) -> None:
        """cam_id is stored from constructor."""
        assert frame_source.cam_id == TEST_CAM_ID

    def test_video_path_constructed(self, frame_source: FrameSource) -> None:
        """video_path is constructed from directory and cam_id."""
        expected_path = TEST_VIDEO_DIR / f"cam_{TEST_CAM_ID}.mp4"
        assert frame_source.video_path == expected_path


class TestSequentialReading:
    """Test read_frame() sequential access."""

    def test_next_frame_returns_frame_packet(self, frame_source: FrameSource) -> None:
        """next_frame() returns a FramePacket."""
        result = frame_source.next_frame()
        assert result is not None
        assert result.frame_index == 0
        assert isinstance(result.frame_time, float)
        assert isinstance(result.frame, np.ndarray)
        assert result.frame.ndim == 3
        assert result.frame.shape[2] == 3

    def test_next_frame_matches_video_size(self, frame_source: FrameSource) -> None:
        """Frame dimensions match video size."""
        result = frame_source.next_frame()
        assert result is not None
        height, width, _ = result.frame.shape
        expected_width, expected_height = frame_source.size
        assert width == expected_width
        assert height == expected_height

    def test_next_frame_returns_none_at_eof(self, frame_source: FrameSource) -> None:
        """next_frame() returns None when video is exhausted."""
        frame_count = 0
        while frame_source.next_frame() is not None:
            frame_count += 1

        assert frame_source.next_frame() is None
        assert frame_count == frame_source.frame_count

    def test_sequential_frames_are_different(self, frame_source: FrameSource) -> None:
        """Sequential frames should be different (video has motion)."""
        r1 = frame_source.next_frame()
        r2 = frame_source.next_frame()
        assert r1 is not None
        assert r2 is not None
        diff = np.abs(r1.frame.astype(np.int16) - r2.frame.astype(np.int16))
        assert np.max(diff) > 0, "Sequential frames should differ"


class TestContextManager:
    """Test context manager protocol."""

    def test_context_manager_opens_and_closes(self) -> None:
        """Context manager properly opens and closes resources."""
        with FrameSource(TEST_VIDEO_DIR, TEST_CAM_ID) as source:
            result = source.next_frame()
            assert result is not None

    def test_context_manager_closes_on_exception(self) -> None:
        """Context manager closes resources even on exception."""
        try:
            with FrameSource(TEST_VIDEO_DIR, TEST_CAM_ID) as source:
                _ = source.next_frame()
                raise ValueError("Test exception")
        except ValueError:
            pass
        # Source should be closed - _container should be None
        assert source._container is None


class TestResourceManagement:
    """Test resource lifecycle and cleanup."""

    def test_close_releases_resources(self, frame_source: FrameSource) -> None:
        """close() releases video container resources."""
        frame_source.close()
        assert frame_source._container is None

    def test_close_is_idempotent(self, frame_source: FrameSource) -> None:
        """close() can be called multiple times safely."""
        frame_source.close()
        frame_source.close()  # Should not raise
        assert frame_source._container is None

    def test_read_after_close_returns_none(self, frame_source: FrameSource) -> None:
        """Operations after close() return None gracefully."""
        frame_source.close()
        assert frame_source.next_frame() is None


class TestInvalidInput:
    """Test handling of invalid inputs."""

    def test_nonexistent_directory_raises(self) -> None:
        """Opening non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            FrameSource(Path("/nonexistent/directory"), 0)

    def test_nonexistent_port_raises(self) -> None:
        """Opening non-existent camera raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            FrameSource(TEST_VIDEO_DIR, 999)


if __name__ == "__main__":
    """Debug harness for running tests with debugpy."""
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Test video dir: {TEST_VIDEO_DIR}")
    print(f"Test camera: {TEST_CAM_ID}")
    print(f"Dir exists: {TEST_VIDEO_DIR.exists()}")

    with FrameSource(TEST_VIDEO_DIR, TEST_CAM_ID) as source:
        print(f"Frame count: {source.frame_count}")
        print(f"FPS: {source.fps}")
        print(f"Size: {source.size}")
        print(f"Camera ID: {source.cam_id}")
        print(f"Start index: {source.start_frame_index}")
        print(f"Last index: {source.last_frame_index}")

        result = source.next_frame()
        print(f"First frame shape: {result.frame.shape if result is not None else None}")
