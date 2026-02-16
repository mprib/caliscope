"""Frame synchronization for multi-camera recordings."""

from pathlib import Path
import pandas as pd

# Type alias: sync_index → {cam_id: frame_index} (None if frame dropped)
SyncMap = dict[int, dict[int, int | None]]


def compute_sync_indices(timestamps_csv: Path) -> SyncMap:
    """Compute sync index assignments from frame timestamps.

    Uses greedy forward-pass with cross-camera interdependencies,
    replicating the logic from the real-time Synchronizer class.

    Args:
        timestamps_csv: CSV with columns (cam_id, frame_time).
            May also have sync_index column (used for validation, not computation).

    Returns:
        Mapping of sync_index to {cam_id: frame_index}.
        frame_index is None if that camera dropped the frame.
    """
    # Load timestamps and organize by cam_id
    df = pd.read_csv(timestamps_csv)

    # Group frames by cam_id, keeping them in temporal order
    frames_by_cam: dict[int, list[float]] = {}
    for cam_key, group in df.groupby("cam_id"):
        # Sort by frame_time to ensure temporal order
        sorted_group = group.sort_values("frame_time")
        # cam_key is numpy.int64 from pandas groupby; convert to Python int
        cam_id = int(sorted_group["cam_id"].iloc[0])
        frames_by_cam[cam_id] = sorted_group["frame_time"].tolist()

    cam_ids = sorted(frames_by_cam.keys())

    # Initialize cursors at frame 0 for each camera
    cursors = {cam_id: 0 for cam_id in cam_ids}

    sync_map: SyncMap = {}
    sync_index = 0

    # Continue until all cameras are exhausted
    while any(cursors[c] < len(frames_by_cam[c]) for c in cam_ids):
        # Collect current candidate frames from each camera
        candidates: dict[int, float] = {}
        for cam_id in cam_ids:
            if cursors[cam_id] < len(frames_by_cam[cam_id]):
                frame_time = frames_by_cam[cam_id][cursors[cam_id]]
                candidates[cam_id] = frame_time

        if not candidates:
            break

        # Build earliest_next and latest_current for each camera
        # (replicating synchronizer.py:245-250)
        earliest_next: dict[int, float] = {}
        latest_current: dict[int, float] = {}

        for cam_id in cam_ids:
            earliest_next[cam_id] = _earliest_next_frame(cam_id, cursors, frames_by_cam)
            latest_current[cam_id] = _latest_current_frame(cam_id, cursors, frames_by_cam)

        # Decide which frames to assign to this sync group
        # (replicating synchronizer.py:253-280)
        assigned: dict[int, int | None] = {}
        for cam_id in cam_ids:
            if cam_id not in candidates:
                # Camera exhausted, mark as None (dropped frame)
                assigned[cam_id] = None
                continue

            frame_time = candidates[cam_id]
            current_frame_index = cursors[cam_id]

            # Skip if frame_time > earliest_next (temporal violation)
            if frame_time > earliest_next[cam_id]:
                assigned[cam_id] = None
                continue

            # Skip if frame is closer to next group than current group
            delta_to_next = earliest_next[cam_id] - frame_time
            delta_to_current = frame_time - latest_current[cam_id]

            if delta_to_next < delta_to_current:
                assigned[cam_id] = None
                continue

            # Assign frame to current sync group and advance cursor
            assigned[cam_id] = current_frame_index
            cursors[cam_id] += 1

        # Only record non-empty sync groups
        # If no frames were assigned, advance the slowest camera to prevent infinite loop
        if any(v is not None for v in assigned.values()):
            sync_map[sync_index] = assigned
            sync_index += 1
        else:
            # No frames assigned - advance the camera with the smallest frame_time
            # to prevent infinite loop (this shouldn't happen in practice with correct data)
            if candidates:
                min_cam = min(candidates.keys(), key=lambda c: candidates[c])
                cursors[min_cam] += 1

    return sync_map


def _earliest_next_frame(cam_id: int, cursors: dict[int, int], frames_by_cam: dict[int, list[float]]) -> float:
    """Get minimum frame_time of NEXT frames from OTHER cameras.

    Replicates synchronizer.py:169-199 (earliest_next_frame method).
    """
    times_of_next_frames = []

    for c in cursors:
        if c == cam_id:
            continue

        next_index = cursors[c] + 1
        if next_index < len(frames_by_cam[c]):
            next_frame_time = frames_by_cam[c][next_index]
            times_of_next_frames.append(next_frame_time)

    # If all other cameras are exhausted, return infinity
    if not times_of_next_frames:
        return float("inf")

    return min(times_of_next_frames)


def _latest_current_frame(cam_id: int, cursors: dict[int, int], frames_by_cam: dict[int, list[float]]) -> float:
    """Get maximum frame_time of CURRENT frames from OTHER cameras.

    Replicates synchronizer.py:201-211 (latest_current_frame method).
    """
    times_of_current_frames = []

    for c in cursors:
        if c == cam_id:
            continue

        current_index = cursors[c]
        if current_index < len(frames_by_cam[c]):
            current_frame_time = frames_by_cam[c][current_index]
            times_of_current_frames.append(current_frame_time)

    # If all other cameras are exhausted, return negative infinity
    if not times_of_current_frames:
        return float("-inf")

    return max(times_of_current_frames)
