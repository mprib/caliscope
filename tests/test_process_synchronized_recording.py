"""Tests for process_synchronized_recording.

Verifies the batch processing function correctly extracts 2D landmarks
from synchronized multi-camera video recordings.
"""

from pathlib import Path

import pytest

from caliscope import persistence
from caliscope.core.process_synchronized_recording import (
    FrameData,
    get_initial_thumbnails,
    process_synchronized_recording,
)
from caliscope.task_manager.cancellation import CancellationToken
from caliscope.trackers.charuco_tracker import CharucoTracker

# Test session with 4 cameras and charuco calibration data
TEST_SESSION = Path("tests/sessions/4_cam_recording")
RECORDING_DIR = TEST_SESSION / "calibration" / "extrinsic"


@pytest.fixture
def cameras():
    """Load camera array from test session."""
    camera_array = persistence.load_camera_array(TEST_SESSION / "camera_array.toml")
    return camera_array.cameras


@pytest.fixture
def tracker():
    """Create charuco tracker from test session config."""
    charuco = persistence.load_charuco(TEST_SESSION / "charuco.toml")
    return CharucoTracker(charuco)


class TestProcessSynchronizedRecording:
    """Tests for process_synchronized_recording function."""

    def test_produces_image_points(self, cameras, tracker):
        """Verify function returns valid ImagePoints with tracked data."""
        # Process a small subset to keep test fast
        image_points = process_synchronized_recording(
            RECORDING_DIR,
            cameras,
            tracker,
            subsample=50,  # Only process every 50th frame for speed
        )

        # Should have detected points
        assert len(image_points.df) > 0

        # Required columns should be present
        required_columns = [
            "sync_index",
            "port",
            "point_id",
            "img_loc_x",
            "img_loc_y",
        ]
        for col in required_columns:
            assert col in image_points.df.columns

        # Should have data from multiple ports
        ports_in_data = image_points.df["port"].unique()
        assert len(ports_in_data) > 1

    def test_subsample_reduces_processed_frames(self, cameras, tracker):
        """Verify subsample parameter reduces frames processed proportionally."""
        # Process every 50th frame (fast)
        all_50 = process_synchronized_recording(
            RECORDING_DIR,
            cameras,
            tracker,
            subsample=50,
        )

        # Process every 100th frame (even faster)
        all_100 = process_synchronized_recording(
            RECORDING_DIR,
            cameras,
            tracker,
            subsample=100,
        )

        # Should have roughly 2x difference in unique sync indices
        syncs_50 = all_50.df["sync_index"].nunique()
        syncs_100 = all_100.df["sync_index"].nunique()

        # Allow for some flexibility due to rounding
        assert syncs_50 >= syncs_100
        assert syncs_50 <= syncs_100 * 2 + 5  # Allow small margin

    def test_progress_callback_invoked(self, cameras, tracker):
        """Verify progress callback is called during processing."""
        progress_calls: list[tuple[int, int]] = []

        def on_progress(current: int, total: int) -> None:
            progress_calls.append((current, total))

        process_synchronized_recording(
            RECORDING_DIR,
            cameras,
            tracker,
            subsample=100,  # Fast
            on_progress=on_progress,
        )

        # Should have been called
        assert len(progress_calls) > 0

        # First call should be (1, total)
        assert progress_calls[0][0] == 1

        # All calls should have same total
        totals = {t for _, t in progress_calls}
        assert len(totals) == 1

        # Last call's current should equal total
        last_current, last_total = progress_calls[-1]
        assert last_current == last_total

    def test_frame_data_callback_invoked(self, cameras, tracker):
        """Verify frame_data callback receives frame data for each sync index."""
        frame_data_calls: list[tuple[int, dict[int, FrameData]]] = []

        def on_frame_data(sync_index: int, data: dict[int, FrameData]) -> None:
            frame_data_calls.append((sync_index, data))

        process_synchronized_recording(
            RECORDING_DIR,
            cameras,
            tracker,
            subsample=100,  # Fast
            on_frame_data=on_frame_data,
        )

        # Should have been called
        assert len(frame_data_calls) > 0

        # Each call should have frame data with valid structure
        for sync_index, data in frame_data_calls[:5]:  # Check first 5
            assert isinstance(sync_index, int)
            assert isinstance(data, dict)
            # Should have at least some ports with data
            assert len(data) > 0
            for port, frame_data in data.items():
                assert isinstance(port, int)
                assert frame_data.frame is not None
                assert frame_data.frame.ndim == 3  # BGR image


class TestCancellation:
    """Tests for cancellation support."""

    def test_cancellation_stops_processing(self, cameras, tracker):
        """Verify CancellationToken stops processing gracefully."""
        token = CancellationToken()
        frames_seen: list[int] = []

        def on_frame_data(sync_index: int, data: dict[int, FrameData]) -> None:
            frames_seen.append(sync_index)
            if len(frames_seen) >= 5:
                token.cancel()

        process_synchronized_recording(
            RECORDING_DIR,
            cameras,
            tracker,
            subsample=10,  # Process more frames to see cancellation
            on_frame_data=on_frame_data,
            token=token,
        )

        # Should have stopped early (well before processing all frames)
        assert len(frames_seen) < 20  # Would be many more without cancellation


class TestGetInitialThumbnails:
    """Tests for get_initial_thumbnails function."""

    def test_returns_frame_for_each_camera(self, cameras):
        """Verify thumbnail extraction returns frame for each camera."""
        thumbnails = get_initial_thumbnails(RECORDING_DIR, cameras)

        # Should have thumbnails for all cameras with videos
        assert len(thumbnails) > 0

        # Each thumbnail should be a valid BGR image
        for port, thumb in thumbnails.items():
            assert thumb.ndim == 3
            assert thumb.shape[2] == 3  # BGR channels


if __name__ == "__main__":
    # Debug harness
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Load fixtures manually
    camera_array = persistence.load_camera_array(TEST_SESSION / "camera_array.toml")
    cams = camera_array.cameras
    charuco = persistence.load_charuco(TEST_SESSION / "charuco.toml")
    trk = CharucoTracker(charuco)

    # Run a basic test
    print("Testing process_synchronized_recording...")
    image_points = process_synchronized_recording(
        RECORDING_DIR,
        cams,
        trk,
        subsample=50,
    )
    print(f"Found {len(image_points.df)} point observations")
    print(f"Unique sync indices: {image_points.df['sync_index'].nunique()}")
    print(f"Ports in data: {sorted(image_points.df['port'].unique())}")

    # Save for inspection
    output_path = debug_dir / "process_sync_recording_output.csv"
    image_points.df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
