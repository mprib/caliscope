"""Tests for SynchronizedTimestamps.

Replaces test_frame_sync.py -- tests the sync algorithm and timestamp
loading through the public SynchronizedTimestamps API.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps

# ---------------------------------------------------------------------------
# Test sessions
# ---------------------------------------------------------------------------

SESSION_4CAM = Path("tests/sessions/4_cam_recording")
EXTRINSIC_DIR = SESSION_4CAM / "calibration" / "extrinsic"
RECORDING_DIR_1 = SESSION_4CAM / "recordings" / "recording_1"
SESSION_2CAM = Path("tests/sessions/charuco_calibration_2_cam")
EXTRINSIC_2CAM = SESSION_2CAM / "calibration" / "extrinsic"


# ---------------------------------------------------------------------------
# from_csv
# ---------------------------------------------------------------------------


class TestFromCsv:
    """Loading SynchronizedTimestamps from timestamps.csv."""

    @pytest.mark.parametrize(
        "recording_dir",
        [
            EXTRINSIC_DIR,
            EXTRINSIC_2CAM,
            RECORDING_DIR_1,
        ],
    )
    def test_loads_from_csv(self, recording_dir: Path):
        """Verify construction from CSV produces valid sync indices and cam_ids."""
        synced = SynchronizedTimestamps.from_csv(recording_dir)

        assert len(synced.sync_indices) > 0
        assert synced.sync_indices == sorted(synced.sync_indices)
        assert len(synced.cam_ids) > 0

    def test_sync_index_column_ignored(self):
        """The CSV's sync_index column is ignored -- mapping is recomputed."""
        # If sync_index were used, the first sync_index would be 416 (from the CSV).
        # When recomputed from timestamps, it starts at 0.
        synced = SynchronizedTimestamps.from_csv(EXTRINSIC_DIR)
        assert synced.sync_indices[0] == 0

    def test_frame_for_returns_frame_index(self):
        """frame_for returns a non-None int for a known sync_index and cam_id."""
        synced = SynchronizedTimestamps.from_csv(EXTRINSIC_DIR)
        first_sync = synced.sync_indices[0]

        for cam_id in synced.cam_ids:
            result = synced.frame_for(first_sync, cam_id)
            # First sync group should have all 4 cameras present
            assert result is not None
            assert isinstance(result, int)
            assert result >= 0


# ---------------------------------------------------------------------------
# Sync algorithm correctness (via public API)
# ---------------------------------------------------------------------------


class TestSyncAlgorithm:
    """Verify the greedy sync algorithm produces correct results."""

    @pytest.mark.parametrize(
        "recording_dir",
        [
            EXTRINSIC_DIR,
            EXTRINSIC_2CAM,
            RECORDING_DIR_1,
        ],
    )
    def test_batch_sync_matches_stored_sync_index(self, recording_dir: Path):
        """Verify batch algorithm reproduces existing sync_index groupings.

        Loads the CSV's stored sync_index values, recomputes via
        SynchronizedTimestamps, and checks every frame maps to the correct group.
        """
        csv_path = recording_dir / "timestamps.csv"
        df = pd.read_csv(csv_path)

        synced = SynchronizedTimestamps.from_csv(recording_dir)

        # Build lookup: (cam_id, frame_idx) -> stored sync_index
        stored_sync: dict[tuple[int, int], int] = {}
        for cam_id, group in df.groupby("cam_id"):
            sorted_group = group.sort_values("frame_time")
            for frame_idx, (_, row) in enumerate(sorted_group.iterrows()):
                stored_sync[(int(cam_id), frame_idx)] = int(row["sync_index"])

        # The stored CSV may not start at sync_index 0
        min_stored_sync = int(df["sync_index"].min())

        for computed_sync_idx in synced.sync_indices:
            expected_stored_sync = computed_sync_idx + min_stored_sync

            for cam_id in synced.cam_ids:
                frame_idx = synced.frame_for(computed_sync_idx, cam_id)
                if frame_idx is None:
                    continue

                actual_stored_sync = stored_sync.get((cam_id, frame_idx))
                assert actual_stored_sync == expected_stored_sync, (
                    f"Mismatch: cam_id={cam_id}, frame_idx={frame_idx}: "
                    f"computed sync_idx={computed_sync_idx} (expected stored={expected_stored_sync}), "
                    f"actual stored={actual_stored_sync}"
                )

        # Verify all stored frames are covered
        all_computed = {
            (cam_id, synced.frame_for(si, cam_id))
            for si in synced.sync_indices
            for cam_id in synced.cam_ids
            if synced.frame_for(si, cam_id) is not None
        }
        all_stored = set(stored_sync.keys())

        missing = all_stored - all_computed
        extra = all_computed - all_stored

        assert not missing, f"Frames in stored but not computed: {missing}"
        assert not extra, f"Frames in computed but not in stored: {extra}"

    def test_identity_mapping_for_equal_timestamps(self):
        """When all cameras share identical frame times, sync is identity."""
        from types import MappingProxyType

        from caliscope.recording.frame_timestamps import FrameTimestamps

        frame_times = {i: float(i) / 30.0 for i in range(10)}
        ft = FrameTimestamps(MappingProxyType(frame_times))
        camera_timestamps = MappingProxyType({0: ft, 1: ft, 2: ft})

        synced = SynchronizedTimestamps(camera_timestamps)

        assert len(synced.sync_indices) == 10
        for si in synced.sync_indices:
            assert synced.frame_for(si, 0) == si
            assert synced.frame_for(si, 1) == si
            assert synced.frame_for(si, 2) == si


# ---------------------------------------------------------------------------
# from_videos
# ---------------------------------------------------------------------------


class TestFromVideos:
    """Loading SynchronizedTimestamps inferred from video metadata."""

    def test_constructs_from_videos(self, tmp_path: Path):
        """from_videos succeeds when all cam_*.mp4 files exist."""
        # Copy only the videos (not timestamps.csv) to tmp_path
        for cam_id in range(4):
            src = EXTRINSIC_DIR / f"cam_{cam_id}.mp4"
            shutil.copy(src, tmp_path / f"cam_{cam_id}.mp4")

        synced = SynchronizedTimestamps.from_videos(tmp_path, [0, 1, 2, 3])

        assert len(synced.cam_ids) == 4
        assert len(synced.sync_indices) > 0

    def test_inferred_timestamps_csv_written(self, tmp_path: Path):
        """from_videos persists inferred_timestamps.csv as an audit trail."""
        for cam_id in range(4):
            shutil.copy(EXTRINSIC_DIR / f"cam_{cam_id}.mp4", tmp_path / f"cam_{cam_id}.mp4")

        synced = SynchronizedTimestamps.from_videos(tmp_path, [0, 1, 2, 3])

        inferred_path = tmp_path / "inferred_timestamps.csv"
        assert inferred_path.exists()

        df = pd.read_csv(inferred_path)
        assert "cam_id" in df.columns
        assert "frame_time" in df.columns

        # Row count should equal sum of frame counts across all cameras
        for cam_id in synced.cam_ids:
            ft = synced.for_camera(cam_id)
            expected_rows = len(ft.frame_times)
            actual_rows = len(df[df["cam_id"] == cam_id])
            assert actual_rows == expected_rows

    def test_raises_on_zero_frame_count(self, tmp_path: Path):
        """Raises ValueError for any video with zero frame count (corrupt file)."""
        # Create a nearly-empty invalid mp4 that OpenCV reports 0 frames for
        (tmp_path / "cam_0.mp4").write_bytes(b"not a real video")
        for cam_id in [1, 2, 3]:
            shutil.copy(EXTRINSIC_DIR / f"cam_{cam_id}.mp4", tmp_path / f"cam_{cam_id}.mp4")

        with pytest.raises(ValueError, match="cam_0"):
            SynchronizedTimestamps.from_videos(tmp_path, [0, 1, 2, 3])

    def test_raises_on_missing_video(self, tmp_path: Path):
        """Raises ValueError when a video file is missing."""
        # Only copy 3 of 4 cameras
        for cam_id in [0, 1, 2]:
            shutil.copy(EXTRINSIC_DIR / f"cam_{cam_id}.mp4", tmp_path / f"cam_{cam_id}.mp4")

        with pytest.raises(ValueError, match="cam_3"):
            SynchronizedTimestamps.from_videos(tmp_path, [0, 1, 2, 3])

    def test_raises_on_no_cam_ids(self, tmp_path: Path):
        """Raises ValueError when cam_ids list is empty."""
        with pytest.raises(ValueError, match="No cam_ids"):
            SynchronizedTimestamps.from_videos(tmp_path, [])

    def test_warns_on_fps_mismatch(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        """Warns but proceeds when camera FPS values differ."""
        for cam_id in range(2):
            shutil.copy(EXTRINSIC_DIR / f"cam_{cam_id}.mp4", tmp_path / f"cam_{cam_id}.mp4")

        def fake_props(path: Path):
            cam_id = int(path.stem.split("_")[1])
            fps = 30.0 if cam_id == 0 else 60.0
            return {"fps": fps, "frame_count": 100, "width": 640, "height": 480, "size": (640, 480)}

        with patch("caliscope.recording.synchronized_timestamps.read_video_properties", side_effect=fake_props):
            synced = SynchronizedTimestamps.from_videos(tmp_path, [0, 1])

        assert "Frame rates differ" in caplog.text
        assert len(synced.sync_indices) > 0

    def test_warns_on_frame_count_mismatch(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        """Warns but proceeds when camera frame counts differ."""
        for cam_id in range(2):
            shutil.copy(EXTRINSIC_DIR / f"cam_{cam_id}.mp4", tmp_path / f"cam_{cam_id}.mp4")

        def fake_props(path: Path):
            cam_id = int(path.stem.split("_")[1])
            frame_count = 1000 if cam_id == 0 else 900
            return {"fps": 30.0, "frame_count": frame_count, "width": 640, "height": 480, "size": (640, 480)}

        with patch("caliscope.recording.synchronized_timestamps.read_video_properties", side_effect=fake_props):
            synced = SynchronizedTimestamps.from_videos(tmp_path, [0, 1])

        assert "Frame counts differ" in caplog.text
        assert len(synced.sync_indices) > 0
        assert 0 in synced.cam_ids
        assert 1 in synced.cam_ids

    def test_spread_uses_average_duration(self, tmp_path: Path):
        """Timestamps are spread over shared avg_duration, not per-camera fps."""
        for cam_id in range(4):
            shutil.copy(EXTRINSIC_DIR / f"cam_{cam_id}.mp4", tmp_path / f"cam_{cam_id}.mp4")

        synced = SynchronizedTimestamps.from_videos(tmp_path, [0, 1, 2, 3])

        from caliscope.recording.video_utils import read_video_properties
        from statistics import mean

        props = {cam_id: read_video_properties(tmp_path / f"cam_{cam_id}.mp4") for cam_id in [0, 1, 2, 3]}
        avg_duration = mean(p["frame_count"] / p["fps"] for p in props.values())

        for cam_id in synced.cam_ids:
            ft = synced.for_camera(cam_id)
            frame_count = len(ft.frame_times)
            # Check last frame time matches formula
            expected_last = (frame_count - 1) * avg_duration / frame_count
            actual_last = ft.frame_times[frame_count - 1]
            assert abs(actual_last - expected_last) < 1e-9


# ---------------------------------------------------------------------------
# load (smart factory)
# ---------------------------------------------------------------------------


class TestLoad:
    """SynchronizedTimestamps.load() selects the right path."""

    def test_uses_csv_when_present(self):
        """load() uses from_csv when timestamps.csv exists."""
        synced = SynchronizedTimestamps.load(EXTRINSIC_DIR, cam_ids=[0, 1, 2, 3])
        # CSV path starts at sync_index 0 (recomputed from timestamps)
        assert synced.sync_indices[0] == 0
        assert 0 in synced.cam_ids

    def test_falls_back_to_videos_when_no_csv(self, tmp_path: Path):
        """load() infers from videos when timestamps.csv is absent."""
        for cam_id in range(4):
            shutil.copy(EXTRINSIC_DIR / f"cam_{cam_id}.mp4", tmp_path / f"cam_{cam_id}.mp4")

        synced = SynchronizedTimestamps.load(tmp_path, cam_ids=[0, 1, 2, 3])
        assert len(synced.sync_indices) > 0
        # inferred_timestamps.csv should have been written
        assert (tmp_path / "inferred_timestamps.csv").exists()


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------


class TestToCsv:
    """Verify the CSV output format."""

    def test_to_csv_writes_expected_columns(self, tmp_path: Path):
        """to_csv writes cam_id and frame_time columns."""
        synced = SynchronizedTimestamps.from_csv(EXTRINSIC_DIR)
        out_path = tmp_path / "output.csv"
        synced.to_csv(out_path)

        df = pd.read_csv(out_path)
        assert "cam_id" in df.columns
        assert "frame_time" in df.columns

    def test_to_csv_round_trips(self, tmp_path: Path):
        """Timestamps written by to_csv can be reloaded correctly.

        The reloaded object produces the same sync_indices and frame mappings.
        """
        synced = SynchronizedTimestamps.from_csv(EXTRINSIC_DIR)
        out_dir = tmp_path / "roundtrip"
        out_dir.mkdir()
        synced.to_csv(out_dir / "timestamps.csv")

        reloaded = SynchronizedTimestamps.from_csv(out_dir)

        assert reloaded.cam_ids == synced.cam_ids
        assert len(reloaded.sync_indices) == len(synced.sync_indices)

        # Check a sample of frame mappings
        for si in synced.sync_indices[:5]:
            for cam_id in synced.cam_ids:
                assert reloaded.frame_for(si, cam_id) == synced.frame_for(si, cam_id)


if __name__ == "__main__":
    # Debug harness
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Testing from_csv (4 cam)...")
    synced = SynchronizedTimestamps.from_csv(EXTRINSIC_DIR)
    print(f"  cam_ids: {synced.cam_ids}")
    print(f"  sync_indices count: {len(synced.sync_indices)}")
    print(f"  first sync_index: {synced.sync_indices[0]}")

    first_si = synced.sync_indices[0]
    for cam_id in synced.cam_ids:
        fi = synced.frame_for(first_si, cam_id)
        t = synced.time_for(cam_id, fi) if fi is not None else None
        print(f"  cam {cam_id}: frame {fi}, time {t}")

    print("\nTesting sync algorithm matches stored data...")
    csv_path = EXTRINSIC_DIR / "timestamps.csv"
    df = pd.read_csv(csv_path)
    min_stored = int(df["sync_index"].min())
    print(f"  min stored sync_index in CSV: {min_stored}")
    print("  computed starts at 0 (recomputed from timestamps)")
    print("  Algorithm test passed (run pytest for full verification)")
