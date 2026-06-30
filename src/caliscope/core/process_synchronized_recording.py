"""Batch processing of synchronized multi-camera video.

Pure function that extracts 2D landmarks from synchronized video streams.
Uses batch synchronization from SynchronizedTimestamps -- no real-time streaming.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraData
from caliscope.core.point_data import ImagePoints
from caliscope.packets import PointPacket
from caliscope.recording.frame_source import FrameSource
from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps
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
    synced_timestamps: SynchronizedTimestamps,
    *,
    subsample: int = 1,
    parallel: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    on_frame_data: Callable[[int, dict[int, FrameData]], None] | None = None,
    token: CancellationToken | None = None,
) -> ImagePoints:
    """Process synchronized video recordings to extract 2D landmarks.

    Each camera decodes forward in its own thread, yielding only the frames
    aligned by SynchronizedTimestamps. A single consumer thread walks sync
    indices in order and assembles cross-camera packets for the live display
    callback. Bounded queues provide backpressure so memory stays flat.
    """
    all_sync_indices = synced_timestamps.sync_indices[::subsample]
    total = len(all_sync_indices)
    cam_ids = [c for c in synced_timestamps.cam_ids if (recording_dir / f"cam_{c}.mp4").exists()]

    logger.info(
        f"Processing {total} sync indices "
        f"(subsample={subsample}, total available={len(synced_timestamps.sync_indices)})"
    )

    # Build per-camera work: sync_index -> frame_index mapping and wanted set.
    cam_work: dict[int, dict[int, int]] = {}  # cam_id -> {frame_index: sync_index}
    for cam_id in cam_ids:
        frame_to_sync: dict[int, int] = {}
        for sync_index in all_sync_indices:
            frame_index = synced_timestamps.frame_for(sync_index, cam_id)
            if frame_index is not None:
                frame_to_sync[frame_index] = sync_index
        cam_work[cam_id] = frame_to_sync

    decode_threads = max(1, (os.cpu_count() or 4) // max(1, len(cam_ids)))
    QUEUE_DEPTH = 8

    # Per-camera bounded queue: producer thread decodes + tracks, pushes results.
    cam_queues: dict[int, Queue[tuple[int, FrameData] | None]] = {
        cam_id: Queue(maxsize=QUEUE_DEPTH) for cam_id in cam_ids
    }

    def _camera_worker(cam_id: int) -> None:
        """Decode forward, track, push (sync_index, FrameData) into the queue."""
        frame_to_sync = cam_work[cam_id]
        camera = cameras[cam_id]
        q = cam_queues[cam_id]

        source = FrameSource(
            recording_dir,
            cam_id,
            decode_threads=decode_threads,
            wanted_indices=set(frame_to_sync),
            pixel_format=tracker.pixel_format,
        )
        try:
            while True:
                if token is not None and token.is_cancelled:
                    break
                raw = source.next_frame()
                if raw is None:
                    break
                sync_index = frame_to_sync[raw.frame_index]
                points = tracker.get_points(raw.frame, cam_id, camera.rotation_count)
                q.put((sync_index, FrameData(raw.frame, points, raw.frame_index)))
        finally:
            source.close()
            q.put(None)  # sentinel

    # Start per-camera decode threads.
    threads: list[Thread] = []
    for cam_id in cam_ids:
        t = Thread(target=_camera_worker, args=(cam_id,), daemon=True)
        t.start()
        threads.append(t)

    # Consumer: walk sync indices in order, pull each camera's matching result.
    # Each camera's queue delivers results in sync-index order (frame indices
    # increase monotonically with sync index, and next_frame is forward-only).
    point_rows: list[dict] = []
    cam_buffers: dict[int, tuple[int, FrameData] | None] = {cam_id: None for cam_id in cam_ids}
    cam_done: set[int] = set()

    def _pull(cam_id: int) -> tuple[int, FrameData] | None:
        """Get the next result for a camera, buffering one-ahead."""
        if cam_buffers[cam_id] is not None:
            return cam_buffers[cam_id]
        item = cam_queues[cam_id].get()
        if item is None:
            cam_done.add(cam_id)
            return None
        cam_buffers[cam_id] = item
        return item

    try:
        for i, sync_index in enumerate(all_sync_indices):
            if token is not None and token.is_cancelled:
                logger.info("Processing cancelled")
                break

            frame_data: dict[int, FrameData] = {}

            for cam_id in cam_ids:
                if cam_id in cam_done:
                    continue
                item = _pull(cam_id)
                if item is None:
                    continue
                item_sync, fd = item
                if item_sync == sync_index:
                    frame_data[cam_id] = fd
                    frame_time = synced_timestamps.time_for(cam_id, fd.frame_index)
                    _accumulate_points(point_rows, sync_index, cam_id, fd.frame_index, frame_time, fd.points)
                    cam_buffers[cam_id] = None  # consumed

            if on_frame_data is not None:
                on_frame_data(sync_index, frame_data)
            if on_progress is not None:
                on_progress(i + 1, total)

    finally:
        # Drain queues so producer threads aren't blocked on put().
        for cam_id in cam_ids:
            if cam_id not in cam_done:
                while True:
                    item = cam_queues[cam_id].get()
                    if item is None:
                        break
        for t in threads:
            t.join(timeout=5.0)

    return _build_image_points(point_rows)


def get_initial_thumbnails(
    recording_dir: Path,
    cameras: dict[int, CameraData],
) -> dict[int, NDArray[np.uint8]]:
    """Extract first frame from each camera for thumbnail display.

    Opens each video briefly with PyAV to decode the first frame,
    then closes immediately. Much faster than FrameSource for this use case.
    """
    import av

    thumbnails: dict[int, NDArray[np.uint8]] = {}

    for cam_id in cameras:
        video_path = recording_dir / f"cam_{cam_id}.mp4"
        if not video_path.exists():
            logger.warning(f"Video file not found for cam_id {cam_id}")
            continue

        try:
            container = av.open(str(video_path))
            try:
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    arr: NDArray[np.uint8] = frame.to_ndarray(format="bgr24")  # type: ignore[assignment]
                    thumbnails[cam_id] = arr
                    break
            finally:
                container.close()
        except Exception as e:
            logger.warning(f"Error reading first frame for cam_id {cam_id}: {e}")

    return thumbnails


def _accumulate_points(
    point_rows: list[dict],
    sync_index: int,
    cam_id: int,
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

    obj_loc_x, obj_loc_y, obj_loc_z = points.obj_loc_list

    for i in range(point_count):
        point_rows.append(
            {
                "sync_index": sync_index,
                "cam_id": cam_id,
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
        df = pd.DataFrame(
            columns=[
                "sync_index",
                "cam_id",
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
