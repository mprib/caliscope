"""Tests for FrameSource - raw video frame access."""

from pathlib import Path

import numpy as np
import pytest

from caliscope.recording.frame_source import FrameSource


# Test video path - use existing test session data
TEST_VIDEO = Path(__file__).parent / "sessions/4_cam_recording/calibration/extrinsic/port_0.mp4"


@pytest.fixture
def frame_source() -> FrameSource:
    """Create a FrameSource for testing."""
    source = FrameSource(TEST_VIDEO)
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

    def test_last_frame_index(self, frame_source: FrameSource) -> None:
        """last_frame_index is frame_count - 1."""
        assert frame_source.last_frame_index == frame_source.frame_count - 1

    def test_video_path_stored(self, frame_source: FrameSource) -> None:
        """video_path is stored for reference."""
        assert frame_source.video_path == TEST_VIDEO


class TestSequentialReading:
    """Test read_frame() sequential access."""

    def test_read_frame_returns_numpy_array(self, frame_source: FrameSource) -> None:
        """read_frame() returns BGR numpy array."""
        frame = frame_source.read_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3  # height, width, channels
        assert frame.shape[2] == 3  # BGR

    def test_read_frame_matches_video_size(self, frame_source: FrameSource) -> None:
        """Frame dimensions match video size."""
        frame = frame_source.read_frame()
        assert frame is not None
        height, width, _ = frame.shape
        expected_width, expected_height = frame_source.size
        assert width == expected_width
        assert height == expected_height

    def test_read_frame_returns_none_at_eof(self, frame_source: FrameSource) -> None:
        """read_frame() returns None when video is exhausted."""
        # Read all frames
        frame_count = 0
        while frame_source.read_frame() is not None:
            frame_count += 1

        # Next read should return None
        assert frame_source.read_frame() is None
        assert frame_count == frame_source.frame_count

    def test_sequential_frames_are_different(self, frame_source: FrameSource) -> None:
        """Sequential frames should be different (video has motion)."""
        frame1 = frame_source.read_frame()
        frame2 = frame_source.read_frame()
        assert frame1 is not None
        assert frame2 is not None
        # Frames should not be identical (unless video is static)
        # Using a tolerance check - at least some pixels should differ
        diff = np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))
        assert np.max(diff) > 0, "Sequential frames should differ"


class TestExactFrameAccess:
    """Test get_frame() exact frame seeking."""

    def test_get_frame_returns_numpy_array(self, frame_source: FrameSource) -> None:
        """get_frame() returns BGR numpy array."""
        frame = frame_source.get_frame(0)
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3

    def test_get_frame_at_start(self, frame_source: FrameSource) -> None:
        """get_frame(0) returns first frame."""
        frame = frame_source.get_frame(0)
        assert frame is not None

    def test_get_frame_at_middle(self, frame_source: FrameSource) -> None:
        """get_frame() works for middle frames."""
        middle = frame_source.frame_count // 2
        frame = frame_source.get_frame(middle)
        assert frame is not None

    def test_get_frame_at_end(self, frame_source: FrameSource) -> None:
        """get_frame() works for last frame."""
        last = frame_source.last_frame_index
        frame = frame_source.get_frame(last)
        assert frame is not None

    def test_get_frame_beyond_end_returns_none(self, frame_source: FrameSource) -> None:
        """get_frame() beyond video length returns None."""
        beyond = frame_source.frame_count + 100
        frame = frame_source.get_frame(beyond)
        assert frame is None

    def test_get_frame_different_positions_return_different_frames(self, frame_source: FrameSource) -> None:
        """get_frame() at different positions returns different frames."""
        frame_start = frame_source.get_frame(0)
        frame_middle = frame_source.get_frame(frame_source.frame_count // 2)
        assert frame_start is not None
        assert frame_middle is not None
        # Should be different frames
        assert not np.array_equal(frame_start, frame_middle)


class TestFastFrameAccess:
    """Test get_frame_fast() keyframe seeking."""

    def test_get_frame_fast_returns_tuple(self, frame_source: FrameSource) -> None:
        """get_frame_fast() returns (frame, actual_index) tuple."""
        result = frame_source.get_frame_fast(10)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, actual_idx = result
        assert frame is not None
        assert isinstance(actual_idx, int)

    def test_get_frame_fast_actual_index_at_or_before_target(self, frame_source: FrameSource) -> None:
        """get_frame_fast() returns keyframe at or before target."""
        target = frame_source.frame_count // 2
        frame, actual_idx = frame_source.get_frame_fast(target)
        assert frame is not None
        # Actual index should be at or before target (keyframe)
        assert actual_idx <= target

    def test_get_frame_fast_returns_valid_frame(self, frame_source: FrameSource) -> None:
        """get_frame_fast() returns valid BGR array."""
        frame, _ = frame_source.get_frame_fast(10)
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3

    def test_get_frame_fast_beyond_end_returns_none_and_minus_one(self, frame_source: FrameSource) -> None:
        """get_frame_fast() beyond video length returns (None, -1).

        Bounds checking prevents PyAV's wrap-around behavior.
        """
        beyond = frame_source.frame_count + 100
        frame, actual_idx = frame_source.get_frame_fast(beyond)
        assert frame is None
        assert actual_idx == -1

    def test_get_frame_fast_negative_index_returns_none_and_minus_one(self, frame_source: FrameSource) -> None:
        """get_frame_fast() with negative index returns (None, -1)."""
        frame, actual_idx = frame_source.get_frame_fast(-1)
        assert frame is None
        assert actual_idx == -1


class TestContextManager:
    """Test context manager protocol."""

    def test_context_manager_opens_and_closes(self) -> None:
        """Context manager properly opens and closes resources."""
        with FrameSource(TEST_VIDEO) as source:
            frame = source.read_frame()
            assert frame is not None

    def test_context_manager_closes_on_exception(self) -> None:
        """Context manager closes resources even on exception."""
        try:
            with FrameSource(TEST_VIDEO) as source:
                _ = source.read_frame()
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
        assert frame_source.read_frame() is None
        assert frame_source.get_frame(0) is None
        frame, idx = frame_source.get_frame_fast(0)
        assert frame is None
        assert idx == -1


class TestInvalidInput:
    """Test handling of invalid inputs."""

    def test_nonexistent_file_raises(self) -> None:
        """Opening non-existent file raises exception."""
        with pytest.raises(Exception):  # av.AVError
            FrameSource(Path("/nonexistent/video.mp4"))

    def test_negative_frame_index_returns_none(self, frame_source: FrameSource) -> None:
        """Negative frame index returns None (bounds checking)."""
        frame = frame_source.get_frame(-1)
        assert frame is None


if __name__ == "__main__":
    """Debug harness for running tests with debugpy."""
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Test basic functionality
    print(f"Test video: {TEST_VIDEO}")
    print(f"Exists: {TEST_VIDEO.exists()}")

    with FrameSource(TEST_VIDEO) as source:
        print(f"Frame count: {source.frame_count}")
        print(f"FPS: {source.fps}")
        print(f"Size: {source.size}")
        print(f"Start index: {source.start_frame_index}")
        print(f"Last index: {source.last_frame_index}")

        # Test sequential read
        frame = source.read_frame()
        print(f"First frame shape: {frame.shape if frame is not None else None}")

        # Test exact seek
        frame = source.get_frame(50)
        print(f"Frame 50 shape: {frame.shape if frame is not None else None}")

        # Test fast seek
        frame, idx = source.get_frame_fast(50)
        print(f"Fast seek to 50: got frame at index {idx}")
