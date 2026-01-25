"""Tests for batch frame synchronization."""

import pandas as pd
import pytest
from pathlib import Path

from caliscope.recording.frame_sync import compute_sync_indices


class TestComputeSyncIndices:
    """Verify batch sync matches existing sync_index values."""

    @pytest.fixture(
        params=[
            "tests/sessions/4_cam_recording/calibration/extrinsic/frame_time_history.csv",
            "tests/sessions/mediapipe_calibration_2_cam/calibration/extrinsic/frame_time_history.csv",
            "tests/sessions/4_cam_recording/recordings/recording_1/frame_time_history.csv",
        ]
    )
    def timestamps_csv(self, request) -> Path:
        return Path(request.param)

    def test_batch_sync_matches_stored_sync_index(self, timestamps_csv: Path):
        """Verify batch algorithm reproduces existing sync_index groupings."""
        df = pd.read_csv(timestamps_csv)

        # Compute using batch algorithm (from frame_time only)
        sync_map = compute_sync_indices(timestamps_csv)

        # Build lookup: (port, frame_idx) -> stored sync_index
        # Frames per port are in temporal order in the CSV
        port_groups = df.groupby("port")
        stored_sync = {}
        for port, group in port_groups:
            sorted_group = group.sort_values("frame_time")
            for frame_idx, (_, row) in enumerate(sorted_group.iterrows()):
                stored_sync[(port, frame_idx)] = row["sync_index"]

        # Determine the offset between computed and stored sync indices
        # The stored CSV may not start at sync_index 0
        min_stored_sync = df["sync_index"].min()
        offset = min_stored_sync  # e.g., if stored starts at 416, offset is 416

        # Verify our computed assignments match the stored groupings
        for sync_idx, port_frames in sync_map.items():
            # Map computed sync_idx to stored sync_idx accounting for offset
            expected_stored_sync = sync_idx + offset

            for port, frame_idx in port_frames.items():
                if frame_idx is None:
                    # Port dropped this frame - verify it's not in stored data for this sync_index
                    # or that all ports have the same sync_index for this group
                    continue

                actual_stored_sync = stored_sync.get((port, frame_idx))
                assert actual_stored_sync == expected_stored_sync, (
                    f"Mismatch: port={port}, frame_idx={frame_idx}: "
                    f"computed sync_idx={sync_idx} (stored={expected_stored_sync}), "
                    f"actual stored={actual_stored_sync}"
                )

        # Also verify that all stored frames were accounted for
        all_computed_frames = set()
        for sync_idx, port_frames in sync_map.items():
            for port, frame_idx in port_frames.items():
                if frame_idx is not None:
                    all_computed_frames.add((port, frame_idx))

        all_stored_frames = set(stored_sync.keys())
        missing_from_computed = all_stored_frames - all_computed_frames
        extra_in_computed = all_computed_frames - all_stored_frames

        assert not missing_from_computed, f"Frames in stored but not computed: {missing_from_computed}"
        assert not extra_in_computed, f"Frames in computed but not stored: {extra_in_computed}"


if __name__ == "__main__":
    # Debug harness
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    test_csvs = [
        Path("tests/sessions/4_cam_recording/calibration/extrinsic/frame_time_history.csv"),
        Path("tests/sessions/mediapipe_calibration_2_cam/calibration/extrinsic/frame_time_history.csv"),
        Path("tests/sessions/4_cam_recording/recordings/recording_1/frame_time_history.csv"),
    ]

    test_instance = TestComputeSyncIndices()

    for csv_path in test_csvs:
        print(f"\nTesting: {csv_path}")
        try:
            test_instance.test_batch_sync_matches_stored_sync_index(csv_path)
            print("  ✓ PASSED")
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")

    print("\nAll tests completed.")
