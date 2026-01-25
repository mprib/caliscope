"""Frame synchronization for multi-camera recordings."""

from pathlib import Path
import pandas as pd

# Type alias: sync_index → {port: frame_index} (None if frame dropped)
SyncMap = dict[int, dict[int, int | None]]


def compute_sync_indices(timestamps_csv: Path) -> SyncMap:
    """Compute sync index assignments from frame timestamps.

    Uses greedy forward-pass with cross-port interdependencies,
    replicating the logic from the real-time Synchronizer class.

    Args:
        timestamps_csv: CSV with columns (port, frame_time).
            May also have sync_index column (used for validation, not computation).

    Returns:
        Mapping of sync_index to {port: frame_index}.
        frame_index is None if that port dropped the frame.
    """
    # Load timestamps and organize by port
    df = pd.read_csv(timestamps_csv)

    # Group frames by port, keeping them in temporal order
    frames_by_port: dict[int, list[float]] = {}
    for port_key, group in df.groupby("port"):
        # Sort by frame_time to ensure temporal order
        sorted_group = group.sort_values("frame_time")
        # port_key is numpy.int64 from pandas groupby; convert to Python int
        port = int(sorted_group["port"].iloc[0])
        frames_by_port[port] = sorted_group["frame_time"].tolist()

    ports = sorted(frames_by_port.keys())

    # Initialize cursors at frame 0 for each port
    cursors = {port: 0 for port in ports}

    sync_map: SyncMap = {}
    sync_index = 0

    # Continue until all ports are exhausted
    while any(cursors[p] < len(frames_by_port[p]) for p in ports):
        # Collect current candidate frames from each port
        candidates: dict[int, float] = {}
        for port in ports:
            if cursors[port] < len(frames_by_port[port]):
                frame_time = frames_by_port[port][cursors[port]]
                candidates[port] = frame_time

        if not candidates:
            break

        # Build earliest_next and latest_current for each port
        # (replicating synchronizer.py:245-250)
        earliest_next: dict[int, float] = {}
        latest_current: dict[int, float] = {}

        for port in ports:
            earliest_next[port] = _earliest_next_frame(port, cursors, frames_by_port)
            latest_current[port] = _latest_current_frame(port, cursors, frames_by_port)

        # Decide which frames to assign to this sync group
        # (replicating synchronizer.py:253-280)
        assigned: dict[int, int | None] = {}
        for port in ports:
            if port not in candidates:
                # Port exhausted, mark as None (dropped frame)
                assigned[port] = None
                continue

            frame_time = candidates[port]
            current_frame_index = cursors[port]

            # Skip if frame_time > earliest_next (temporal violation)
            if frame_time > earliest_next[port]:
                assigned[port] = None
                continue

            # Skip if frame is closer to next group than current group
            delta_to_next = earliest_next[port] - frame_time
            delta_to_current = frame_time - latest_current[port]

            if delta_to_next < delta_to_current:
                assigned[port] = None
                continue

            # Assign frame to current sync group and advance cursor
            assigned[port] = current_frame_index
            cursors[port] += 1

        # Only record non-empty sync groups
        # If no frames were assigned, advance the slowest port to prevent infinite loop
        if any(v is not None for v in assigned.values()):
            sync_map[sync_index] = assigned
            sync_index += 1
        else:
            # No frames assigned - advance the port with the smallest frame_time
            # to prevent infinite loop (this shouldn't happen in practice with correct data)
            if candidates:
                min_port = min(candidates.keys(), key=lambda p: candidates[p])
                cursors[min_port] += 1

    return sync_map


def _earliest_next_frame(port: int, cursors: dict[int, int], frames_by_port: dict[int, list[float]]) -> float:
    """Get minimum frame_time of NEXT frames from OTHER ports.

    Replicates synchronizer.py:169-199 (earliest_next_frame method).
    """
    times_of_next_frames = []

    for p in cursors:
        if p == port:
            continue

        next_index = cursors[p] + 1
        if next_index < len(frames_by_port[p]):
            next_frame_time = frames_by_port[p][next_index]
            times_of_next_frames.append(next_frame_time)

    # If all other ports are exhausted, return infinity
    if not times_of_next_frames:
        return float("inf")

    return min(times_of_next_frames)


def _latest_current_frame(port: int, cursors: dict[int, int], frames_by_port: dict[int, list[float]]) -> float:
    """Get maximum frame_time of CURRENT frames from OTHER ports.

    Replicates synchronizer.py:201-211 (latest_current_frame method).
    """
    times_of_current_frames = []

    for p in cursors:
        if p == port:
            continue

        current_index = cursors[p]
        if current_index < len(frames_by_port[p]):
            current_frame_time = frames_by_port[p][current_index]
            times_of_current_frames.append(current_frame_time)

    # If all other ports are exhausted, return negative infinity
    if not times_of_current_frames:
        return float("-inf")

    return max(times_of_current_frames)
