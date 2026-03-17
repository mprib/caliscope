"""Tests for FrameSource.read_frame_at() sequential read optimization.

Validates that read_frame_at() returns identical frames to get_frame()
across sequential, skip, and large-skip access patterns.
"""

from pathlib import Path
import numpy as np
import pytest
from caliscope.recording.frame_source import FrameSource

TEST_SESSION = Path("tests/sessions/4_cam_recording")
RECORDING_DIR = TEST_SESSION / "calibration" / "extrinsic"


@pytest.fixture
def frame_source():
    """Create FrameSource for cam_0."""
    fs = FrameSource(RECORDING_DIR, cam_id=0)
    yield fs
    fs.close()


@pytest.fixture
def reference_source():
    """Separate FrameSource for get_frame() reference."""
    fs = FrameSource(RECORDING_DIR, cam_id=0)
    yield fs
    fs.close()


class TestReadFrameAt:
    """Tests for sequential read optimization."""

    def test_sequential_matches_get_frame(self, frame_source, reference_source):
        """Sequential read_frame_at() returns same frames as get_frame()."""
        for i in range(10):
            seq_frame = frame_source.read_frame_at(i)
            ref_frame = reference_source.get_frame(i)
            assert seq_frame is not None
            assert ref_frame is not None
            np.testing.assert_array_equal(seq_frame, ref_frame)

    def test_small_skip_matches(self, frame_source, reference_source):
        """Small forward skip returns correct frame."""
        # Read frame 0 to initialize
        frame_source.read_frame_at(0)
        # Skip to frame 5
        seq_frame = frame_source.read_frame_at(5)
        ref_frame = reference_source.get_frame(5)
        assert seq_frame is not None
        assert ref_frame is not None
        np.testing.assert_array_equal(seq_frame, ref_frame)

    def test_large_skip_matches(self, frame_source, reference_source):
        """Large forward skip (> GOP) returns correct frame."""
        # Read frame 0 to initialize
        frame_source.read_frame_at(0)
        # Skip to a frame well beyond GOP size (typically ~30)
        target = min(100, frame_source.last_frame_index)
        seq_frame = frame_source.read_frame_at(target)
        ref_frame = reference_source.get_frame(target)
        assert seq_frame is not None
        assert ref_frame is not None
        np.testing.assert_array_equal(seq_frame, ref_frame)

    def test_backward_seek(self, frame_source, reference_source):
        """Backward seek returns correct frame."""
        frame_source.read_frame_at(50)
        seq_frame = frame_source.read_frame_at(10)
        ref_frame = reference_source.get_frame(10)
        assert seq_frame is not None
        assert ref_frame is not None
        np.testing.assert_array_equal(seq_frame, ref_frame)

    def test_out_of_bounds(self, frame_source):
        """Out-of-bounds indices return None."""
        assert frame_source.read_frame_at(-1) is None
        assert frame_source.read_frame_at(frame_source.last_frame_index + 1) is None

    def test_cold_start(self, frame_source, reference_source):
        """First call without any prior reads works correctly."""
        target = 20
        seq_frame = frame_source.read_frame_at(target)
        ref_frame = reference_source.get_frame(target)
        assert seq_frame is not None
        assert ref_frame is not None
        np.testing.assert_array_equal(seq_frame, ref_frame)


if __name__ == "__main__":
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    fs = FrameSource(RECORDING_DIR, cam_id=0)
    ref = FrameSource(RECORDING_DIR, cam_id=0)

    print(f"Frame count: {fs.frame_count}, last index: {fs.last_frame_index}")
    print(f"Keyframes: {len(fs._keyframe_indices)}")

    # Test sequential reads
    for i in range(5):
        seq = fs.read_frame_at(i)
        ref_f = ref.get_frame(i)
        match = np.array_equal(seq, ref_f)
        print(f"Frame {i}: match={match}")

    fs.close()
    ref.close()
