"""Batch processing of synchronized multi-camera video.

Pure function that extracts 2D landmarks from synchronized video streams.
Uses batch synchronization from frame_time_history.csv — no real-time streaming.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraData
from caliscope.core.point_data import ImagePoints
from caliscope.packets import PointPacket
from caliscope.recording.frame_source import FrameSource
from caliscope.recording.frame_sync import compute_sync_indices
from caliscope.task_manager.cancellation import CancellationToken
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Frame data for a single camera at a sync index."""

    frame: NDArray[np.uint8]
    points: PointPacket | None
    frame_index: int


def process_synchronized_recording(
    recording_dir: Path,
    cameras: dict[int, CameraData],
    tracker: Tracker,
    *,
    subsample: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
    on_frame_data: Callable[[int, dict[int, FrameData]], None] | None = None,
    token: CancellationToken | None = None,
) -> ImagePoints:
    """Process synchronized video recordings to extract 2D landmarks.

    Reads frame_time_history.csv to determine frame alignment, then processes
    each sync index by seeking directly to aligned frames.

    Args:
        recording_dir: Directory containing port_N.mp4 and frame_time_history.csv
        cameras: Camera data by port (provides rotation_count for frame orientation)
        tracker: Tracker for 2D point extraction (handles per-port state internally)
        subsample: Process every Nth sync index (1 = all, 10 = every 10th)
        on_progress: Called with (current, total) during processing
        on_frame_data: Called with (sync_index, {port: FrameData}) for live display
        token: Cancellation token for graceful shutdown

    Returns:
        ImagePoints containing all tracked 2D observations
    """
    # Load frame timestamps and compute sync assignments
    # Note: CSV is named frame_time_history.csv until Task 1.7 renames it
    timestamps_csv = recording_dir / "frame_time_history.csv"
    sync_map = compute_sync_indices(timestamps_csv)

    # Load frame_time data for enriching output
    timestamps_df = pd.read_csv(timestamps_csv)
    frame_times = _build_frame_time_lookup(timestamps_df)

    # Get sync indices to process (with subsampling)
    all_sync_indices = sorted(sync_map.keys())
    sync_indices_to_process = all_sync_indices[::subsample]
    total = len(sync_indices_to_process)

    logger.info(f"Processing {total} sync indices (subsample={subsample}, total available={len(all_sync_indices)})")

    # Create frame sources (one per camera, for seeking)
    frame_sources = _create_frame_sources(recording_dir, cameras)

    # Point accumulation
    point_rows: list[dict] = []

    try:
        for i, sync_index in enumerate(sync_indices_to_process):
            # Check cancellation
            if token is not None and token.is_cancelled:
                logger.info("Processing cancelled")
                break

            # Read and track frames for this sync index
            frame_data: dict[int, FrameData] = {}
            port_assignments = sync_map[sync_index]

            for port, frame_index in port_assignments.items():
                if frame_index is None:
                    logger.debug(f"Dropped frame: sync={sync_index}, port={port}")
                    continue

                if port not in frame_sources:
                    # Camera in sync_map but not in cameras dict (shouldn't happen)
                    logger.warning(f"Port {port} not in cameras dict, skipping")
                    continue

                camera = cameras[port]
                frame = frame_sources[port].get_frame(frame_index)

                if frame is None:
                    logger.warning(f"Failed to read frame: sync={sync_index}, port={port}, frame_index={frame_index}")
                    continue

                # Tracker handles per-port state internally via port parameter
                points = tracker.get_points(frame, port, camera.rotation_count)
                frame_data[port] = FrameData(frame, points, frame_index)

                # Accumulate points
                frame_time = frame_times.get((port, frame_index), 0.0)
                _accumulate_points(point_rows, sync_index, port, frame_index, frame_time, points)

            # Invoke callbacks
            if on_frame_data is not None:
                on_frame_data(sync_index, frame_data)
            if on_progress is not None:
                on_progress(i + 1, total)

    finally:
        for source in frame_sources.values():
            source.close()

    return _build_image_points(point_rows)


def get_initial_thumbnails(
    recording_dir: Path,
    cameras: dict[int, CameraData],
) -> dict[int, NDArray[np.uint8]]:
    """Extract first frame from each camera for thumbnail display.

    Uses same FrameSource mechanism as processing, just reads frame 0.

    Args:
        recording_dir: Directory containing port_N.mp4 files
        cameras: Camera data by port

    Returns:
        Mapping of port -> first frame (BGR image)
    """
    thumbnails: dict[int, NDArray[np.uint8]] = {}

    for port in cameras:
        try:
            source = FrameSource(recording_dir, port)
            frame = source.get_frame(0)
            source.close()

            if frame is not None:
                thumbnails[port] = frame
            else:
                logger.warning(f"Could not read first frame for port {port}")
        except FileNotFoundError:
            logger.warning(f"Video file not found for port {port}")
        except ValueError as e:
            logger.warning(f"Error opening video for port {port}: {e}")

    return thumbnails


def _create_frame_sources(recording_dir: Path, cameras: dict[int, CameraData]) -> dict[int, FrameSource]:
    """Create FrameSource for each camera port."""
    sources: dict[int, FrameSource] = {}

    for port in cameras:
        try:
            sources[port] = FrameSource(recording_dir, port)
        except FileNotFoundError:
            logger.warning(f"Video file not found for port {port}, skipping")
        except ValueError as e:
            logger.warning(f"Error opening video for port {port}: {e}")

    return sources


def _build_frame_time_lookup(timestamps_df: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Build lookup table: (port, frame_index) -> frame_time.

    Frame index is the row number within each port's sequence.
    """
    lookup: dict[tuple[int, int], float] = {}

    for port, group in timestamps_df.groupby("port"):
        sorted_group = group.sort_values("frame_time").reset_index(drop=True)
        for frame_index, row in sorted_group.iterrows():
            # frame_index here is actually the integer index from iterrows
            lookup[(int(port), int(frame_index))] = float(row["frame_time"])  # type: ignore[arg-type]

    return lookup


def _accumulate_points(
    point_rows: list[dict],
    sync_index: int,
    port: int,
    frame_index: int,
    frame_time: float,
    points: PointPacket | None,
) -> None:
    """Append point data to accumulator list."""
    if points is None:
        return

    point_count = len(points.point_id)
    if point_count == 0:
        return

    # Get obj_loc columns (may be None)
    obj_loc_x, obj_loc_y, obj_loc_z = points.obj_loc_list

    for i in range(point_count):
        point_rows.append(
            {
                "sync_index": sync_index,
                "port": port,
                "frame_index": frame_index,
                "frame_time": frame_time,
                "point_id": int(points.point_id[i]),
                "img_loc_x": float(points.img_loc[i, 0]),
                "img_loc_y": float(points.img_loc[i, 1]),
                "obj_loc_x": obj_loc_x[i],
                "obj_loc_y": obj_loc_y[i],
                "obj_loc_z": obj_loc_z[i],
            }
        )


def _build_image_points(point_rows: list[dict]) -> ImagePoints:
    """Construct ImagePoints from accumulated point data."""
    if not point_rows:
        # Return empty ImagePoints with correct schema
        df = pd.DataFrame(
            columns=[
                "sync_index",
                "port",
                "frame_index",
                "frame_time",
                "point_id",
                "img_loc_x",
                "img_loc_y",
                "obj_loc_x",
                "obj_loc_y",
                "obj_loc_z",
            ]
        )
        return ImagePoints(df)

    df = pd.DataFrame(point_rows)
    return ImagePoints(df)
