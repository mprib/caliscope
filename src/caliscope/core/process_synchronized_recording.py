"""Batch processing of synchronized multi-camera video.

Pure function that extracts 2D landmarks from synchronized video streams.
Uses batch synchronization from SynchronizedTimestamps -- no real-time streaming.
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
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

    Uses SynchronizedTimestamps for frame alignment, then processes each
    sync index by seeking directly to aligned frames.

    Args:
        recording_dir: Directory containing cam_N.mp4 files
        cameras: Camera data by cam_id (provides rotation_count for frame orientation)
        tracker: Tracker for 2D point extraction (handles per-cam_id state internally)
        synced_timestamps: Pre-constructed timestamp alignment object
        subsample: Process every Nth sync index (1 = all, 10 = every 10th)
        parallel: Process cameras concurrently (True) or serially (False).
            Uses ThreadPoolExecutor when True and multiple cameras present.
            Set to False as fallback if threading issues are discovered.
        on_progress: Called with (current, total) during processing
        on_frame_data: Called with (sync_index, {cam_id: FrameData}) for live display
        token: Cancellation token for graceful shutdown

    Returns:
        ImagePoints containing all tracked 2D observations
    """
    all_sync_indices = synced_timestamps.sync_indices[::subsample]
    total = len(all_sync_indices)

    logger.info(
        f"Processing {total} sync indices "
        f"(subsample={subsample}, total available={len(synced_timestamps.sync_indices)})"
    )

    frame_sources = _create_frame_sources(recording_dir, cameras)
    point_rows: list[dict] = []

    try:
        use_pool = parallel and len(frame_sources) > 1

        if use_pool:
            camera_pool = ThreadPoolExecutor(max_workers=len(frame_sources))
        else:
            camera_pool = None

        try:
            for i, sync_index in enumerate(all_sync_indices):
                if token is not None and token.is_cancelled:
                    logger.info("Processing cancelled")
                    break

                frame_data: dict[int, FrameData] = {}

                if use_pool and camera_pool is not None:
                    # --- Parallel path ---
                    futures: dict[int, Future[tuple[int, FrameData | None, list[dict]]]] = {}
                    for cam_id in synced_timestamps.cam_ids:
                        frame_index = synced_timestamps.frame_for(sync_index, cam_id)
                        if frame_index is None:
                            continue
                        if cam_id not in frame_sources:
                            continue
                        camera = cameras[cam_id]
                        frame_time = synced_timestamps.time_for(cam_id, frame_index)
                        futures[cam_id] = camera_pool.submit(
                            _process_one_camera,
                            cam_id,
                            sync_index,
                            frame_index,
                            frame_sources[cam_id],
                            camera,
                            tracker,
                            frame_time,
                        )

                    for cam_id, future in futures.items():
                        _, fd, rows = future.result()
                        if fd is not None:
                            frame_data[cam_id] = fd
                        point_rows.extend(rows)
                else:
                    # --- Serial path (original logic) ---
                    for cam_id in synced_timestamps.cam_ids:
                        frame_index = synced_timestamps.frame_for(sync_index, cam_id)
                        if frame_index is None:
                            continue
                        if cam_id not in frame_sources:
                            continue
                        camera = cameras[cam_id]
                        frame = frame_sources[cam_id].read_frame_at(frame_index)
                        if frame is None:
                            logger.warning(
                                f"Failed to read frame: sync={sync_index}, cam_id={cam_id}, frame_index={frame_index}"
                            )
                            continue
                        points = tracker.get_points(frame, cam_id, camera.rotation_count)
                        frame_data[cam_id] = FrameData(frame, points, frame_index)
                        frame_time = synced_timestamps.time_for(cam_id, frame_index)
                        _accumulate_points(point_rows, sync_index, cam_id, frame_index, frame_time, points)

                # Threading contract: callbacks are always invoked from this
                # thread (the worker thread that owns the sync-index loop),
                # never from pool threads. Presenters rely on this guarantee
                # for unsynchronized accumulator state.
                if on_frame_data is not None:
                    on_frame_data(sync_index, frame_data)
                if on_progress is not None:
                    on_progress(i + 1, total)
        finally:
            if camera_pool is not None:
                camera_pool.shutdown(wait=False)

    finally:
        for source in frame_sources.values():
            source.close()

    return _build_image_points(point_rows)


def get_initial_thumbnails(
    recording_dir: Path,
    cameras: dict[int, CameraData],
) -> dict[int, NDArray[np.uint8]]:
    """Extract first frame from each camera for thumbnail display.

    Opens each video briefly with PyAV to decode the first frame,
    then closes immediately. No keyframe scanning or frame index
    construction -- much faster than FrameSource for this use case.

    Args:
        recording_dir: Directory containing cam_N.mp4 files
        cameras: Camera data by cam_id

    Returns:
        Mapping of cam_id -> first frame (BGR image)
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
                    # bgr24 always produces uint8; PyAV stubs don't narrow the type
                    arr: NDArray[np.uint8] = frame.to_ndarray(format="bgr24")  # type: ignore[assignment]
                    thumbnails[cam_id] = arr
                    break
            finally:
                container.close()
        except Exception as e:
            logger.warning(f"Error reading first frame for cam_id {cam_id}: {e}")

    return thumbnails


def _create_frame_sources(recording_dir: Path, cameras: dict[int, CameraData]) -> dict[int, FrameSource]:
    """Create FrameSource for each camera cam_id.

    Each FrameSource runs a keyframe scan on init (I/O-bound), so cameras
    are initialized in parallel threads.
    """

    def _init_one(cam_id: int) -> tuple[int, FrameSource | None]:
        try:
            return cam_id, FrameSource(recording_dir, cam_id)
        except FileNotFoundError:
            logger.warning(f"Video file not found for cam_id {cam_id}, skipping")
            return cam_id, None
        except ValueError as e:
            logger.warning(f"Error opening video for cam_id {cam_id}: {e}")
            return cam_id, None

    cam_ids = list(cameras.keys())

    with ThreadPoolExecutor(max_workers=len(cam_ids)) as pool:
        results = pool.map(_init_one, cam_ids)

    return {cam_id: source for cam_id, source in results if source is not None}


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


def _process_one_camera(
    cam_id: int,
    sync_index: int,
    frame_index: int,
    frame_source: FrameSource,
    camera: CameraData,
    tracker: Tracker,
    frame_time: float,
) -> tuple[int, FrameData | None, list[dict]]:
    """Process a single camera for one sync index.

    Thread safety: This function is safe to call concurrently for different
    cam_ids because:
    - Each FrameSource instance is dedicated to one camera (no sharing).
    - Tracker.get_points() is thread-safe:
      - OnnxTracker._prev_bboxes: keyed by cam_id, each thread accesses
        only its own key. Dict internal structure is GIL-protected.
        INVARIANT: thread safety depends on each thread accessing a
        distinct cam_id. Two threads must never process the same cam_id
        concurrently.
      - OnnxTracker.session.run(): onnxruntime InferenceSession.run() is
        thread-safe (C++ session uses read-only model weights, per-call
        buffer allocation). Verified for CPU execution provider.
      - CharucoTracker/ArUcoTracker/ChessboardTracker: stateless OpenCV
        calls on caller-provided buffers.
    - point_rows is built locally and returned, not shared.

    Returns:
        (cam_id, frame_data_or_none, point_rows)
    """
    frame = frame_source.read_frame_at(frame_index)

    if frame is None:
        logger.warning(f"Failed to read frame: sync={sync_index}, cam_id={cam_id}, frame_index={frame_index}")
        return cam_id, None, []

    points = tracker.get_points(frame, cam_id, camera.rotation_count)
    fd = FrameData(frame, points, frame_index)

    local_rows: list[dict] = []
    _accumulate_points(local_rows, sync_index, cam_id, frame_index, frame_time, points)

    return cam_id, fd, local_rows


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
