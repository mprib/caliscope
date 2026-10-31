"""Box-first person selection for synchronized recordings.

This module keeps detection, camera-local association, and pose estimation as
separate passes.  Boxes are stored in the original video coordinate system so
they can be inspected and persisted without depending on inference rotation.
"""

from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from caliscope.core.box_tracks import BOX_TRACK_COLUMNS, BoxTracks
from caliscope.core.point_data import ImagePoints
from caliscope.recording.frame_source import FrameSource
from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps
from caliscope.reporting import ProgressCallback
from caliscope.trackers.byte_tracker import ByteTracker, LOW_SCORE_THRESHOLD
from caliscope.trackers.helper import apply_rotation, unrotate_points


class PersonDetector(Protocol):
    def people(self, frame: np.ndarray, min_score: float | None = None) -> tuple[np.ndarray, np.ndarray]: ...


class BoxPoseTracker(Protocol):
    pixel_format: Any

    def keypoints_in_box(
        self, frame: np.ndarray, box: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, np.ndarray]: ...


def track_people(
    videos: Mapping[int, Path | str],
    detector: PersonDetector,
    *,
    frame_step: int = 1,
    timestamps: Path | str | None = None,
    rotation_counts: Mapping[int, int] | None = None,
    progress: ProgressCallback | None = None,
) -> BoxTracks:
    """Detect and associate every person box at sampled synchronized moments."""
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")
    video_paths = _video_paths(videos)
    rotations = rotation_counts or {}
    synced = _synchronized_timestamps(video_paths, timestamps)
    selected_sync_indices = synced.sync_indices[::frame_step]
    decode_threads = max(1, (os.cpu_count() or 4) // len(video_paths))

    def process_camera(cam_id: int, path: Path) -> list[dict[str, Any]]:
        work = _sync_work(synced, cam_id, selected_sync_indices)
        if progress is not None:
            progress.on_video_start(cam_id, len(work))
        if not work:
            if progress is not None:
                progress.on_video_complete(cam_id)
            return []

        tracker = ByteTracker()
        sync_for_frame = {frame_index: sync_index for sync_index, frame_index in work}
        source = FrameSource.from_path(
            path, cam_id=cam_id, decode_threads=decode_threads, wanted_indices=set(sync_for_frame)
        )
        rows: list[dict[str, Any]] = []
        try:
            processed = 0
            while (raw := source.next_frame()) is not None:
                rotation_count = rotations.get(cam_id, 0)
                inference_frame = apply_rotation(raw.frame, rotation_count)
                boxes, scores = detector.people(inference_frame, min_score=LOW_SCORE_THRESHOLD)
                for track_id, detected_box, detector_score in tracker.update(boxes, scores):
                    source_box = _unrotate_box(detected_box, rotation_count, raw.frame.shape[1], raw.frame.shape[0])
                    rows.append(
                        {
                            "sync_index": sync_for_frame[raw.frame_index],
                            "cam_id": cam_id,
                            "frame_index": raw.frame_index,
                            "frame_time": synced.time_for(cam_id, raw.frame_index),
                            "track_id": track_id,
                            "box_x1": source_box[0],
                            "box_y1": source_box[1],
                            "box_x2": source_box[2],
                            "box_y2": source_box[3],
                            "detector_score": detector_score,
                            "image_width": raw.frame.shape[1],
                            "image_height": raw.frame.shape[0],
                            "rotation_count": rotation_count,
                        }
                    )
                processed += 1
                if progress is not None:
                    progress.on_frame(cam_id, processed, len(boxes))
        finally:
            source.close()
        if progress is not None:
            progress.on_video_complete(cam_id)
        return rows

    return BoxTracks(pd.DataFrame(_run_per_camera(video_paths, process_camera), columns=list(BOX_TRACK_COLUMNS)))


def estimate_keypoints(
    videos: Mapping[int, Path | str],
    pose_tracker: BoxPoseTracker,
    box_tracks: BoxTracks,
    *,
    min_confidence: float,
    progress: ProgressCallback | None = None,
) -> ImagePoints:
    """Estimate pose only in supplied source-coordinate boxes.

    The recorded rows are the timing authority for this pass.  It does not
    resynchronize videos or infer observations for gaps in the box table.
    """
    # Some supported model decoders expose unnormalised confidence scores, so
    # a hard upper bound would reject otherwise valid tracker output.
    if not np.isfinite(min_confidence) or min_confidence < 0:
        raise ValueError(f"min_confidence must be finite and non-negative, got {min_confidence}")
    boxes = box_tracks.df
    _validate_selected_tracks(boxes)
    table_cameras = set(boxes["cam_id"].unique())
    missing_cameras = sorted(table_cameras - set(videos))
    if missing_cameras:
        raise ValueError(f"BoxTracks contains camera IDs absent from videos: {missing_cameras}")
    video_paths = _video_paths(videos)

    active_cameras = [cam_id for cam_id in video_paths if cam_id in table_cameras]
    if not active_cameras:
        return _empty_image_points()
    decode_threads = max(1, (os.cpu_count() or 4) // len(active_cameras))

    def process_camera(cam_id: int, path: Path) -> list[dict[str, Any]]:
        camera_boxes = boxes.loc[boxes["cam_id"] == cam_id].copy()
        by_frame = _rows_by_frame(camera_boxes, cam_id)
        if progress is not None:
            progress.on_video_start(cam_id, len(by_frame))
        source = FrameSource.from_path(
            path,
            cam_id=cam_id,
            decode_threads=decode_threads,
            wanted_indices=set(by_frame),
            pixel_format=pose_tracker.pixel_format,
        )
        rows: list[dict[str, Any]] = []
        seen_frames: set[int] = set()
        try:
            processed = 0
            while (raw := source.next_frame()) is not None:
                seen_frames.add(raw.frame_index)
                box = by_frame[raw.frame_index]
                rotation_count = int(box.rotation_count)
                _validate_source_geometry(box, raw.frame.shape[1], raw.frame.shape[0], cam_id)
                inference_frame = apply_rotation(raw.frame, rotation_count)
                inference_box = _rotate_box(
                    (box.box_x1, box.box_y1, box.box_x2, box.box_y2),
                    rotation_count,
                    raw.frame.shape[1],
                    raw.frame.shape[0],
                )
                xy, confidence = pose_tracker.keypoints_in_box(inference_frame, inference_box)
                xy = _unrotate_keypoints(xy, rotation_count, inference_frame.shape[1], inference_frame.shape[0])
                keep = _valid_keypoints(xy, confidence, min_confidence, raw.frame.shape[1], raw.frame.shape[0])
                for keypoint_id in np.flatnonzero(keep):
                    rows.append(
                        {
                            "sync_index": int(box.sync_index),
                            "cam_id": cam_id,
                            "frame_index": int(box.frame_index),
                            "frame_time": float(box.frame_time),
                            "object_id": 0,
                            "keypoint_id": int(keypoint_id),
                            "img_loc_x": float(xy[keypoint_id, 0]),
                            "img_loc_y": float(xy[keypoint_id, 1]),
                            "obj_loc_x": np.nan,
                            "obj_loc_y": np.nan,
                            "obj_loc_z": np.nan,
                            "confidence": float(confidence[keypoint_id]),
                            "track_id": int(box.track_id),
                        }
                    )
                processed += 1
                if progress is not None:
                    progress.on_frame(cam_id, processed, int(keep.sum()))
        finally:
            source.close()
        missing_frames = sorted(set(by_frame) - seen_frames)
        if missing_frames:
            shown = ", ".join(str(frame_index) for frame_index in missing_frames[:10])
            suffix = "..." if len(missing_frames) > 10 else ""
            raise ValueError(
                f"BoxTracks references frames unavailable in video for camera {cam_id}: {shown}{suffix}. "
                "Regenerate boxes from this video or select only available frames."
            )
        if progress is not None:
            progress.on_video_complete(cam_id)
        return rows

    rows = _run_per_camera({cam_id: video_paths[cam_id] for cam_id in active_cameras}, process_camera)
    return ImagePoints(pd.DataFrame(rows)) if rows else _empty_image_points()


def _video_paths(videos: Mapping[int, Path | str]) -> dict[int, Path]:
    if not videos:
        raise ValueError("videos must contain at least one camera")
    paths = {int(cam_id): Path(path) for cam_id, path in videos.items()}
    missing = {cam_id: path for cam_id, path in paths.items() if not path.exists()}
    if missing:
        detail = "\n".join(f"  cam {cam_id}: {path}" for cam_id, path in missing.items())
        raise FileNotFoundError(f"Video files not found:\n{detail}")
    return paths


def _synchronized_timestamps(videos: Mapping[int, Path], timestamps: Path | str | None) -> SynchronizedTimestamps:
    if timestamps is not None:
        return SynchronizedTimestamps.from_csv_path(Path(timestamps))
    return SynchronizedTimestamps.from_video_paths(videos)


def _sync_work(synced: SynchronizedTimestamps, cam_id: int, sync_indices: list[int]) -> list[tuple[int, int]]:
    return [
        (sync_index, frame_index)
        for sync_index in sync_indices
        if (frame_index := synced.frame_for(sync_index, cam_id)) is not None
    ]


def _run_per_camera(videos: Mapping[int, Path], work: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(videos), 8)) as executor:
        futures = [executor.submit(work, cam_id, path) for cam_id, path in videos.items()]
        try:
            for future in concurrent.futures.as_completed(futures):
                rows.extend(future.result())
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return rows


def _rows_by_frame(boxes: pd.DataFrame, cam_id: int) -> dict[int, Any]:
    if boxes["frame_index"].duplicated().any():
        raise ValueError(f"BoxTracks has multiple selected boxes for camera {cam_id} at the same frame")
    return {
        int(frame_index): row for frame_index, row in zip(boxes["frame_index"].tolist(), boxes.itertuples(index=False))
    }


def _validate_selected_tracks(boxes: pd.DataFrame) -> None:
    if boxes.empty:
        return
    counts = boxes.groupby("cam_id")["track_id"].nunique()
    multiple = counts[counts > 1]
    if not multiple.empty:
        detail = ", ".join(f"cam {cam_id}: {count} tracks" for cam_id, count in multiple.items())
        raise ValueError(f"estimate_keypoints requires at most one track per camera; found {detail}")


def _valid_keypoints(
    xy: np.ndarray, confidence: np.ndarray, min_confidence: float, width: int, height: int
) -> np.ndarray:
    return (
        np.isfinite(xy).all(axis=1)
        & np.isfinite(confidence)
        & (confidence > 0)
        & (confidence >= min_confidence)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )


def _validate_source_geometry(box: Any, width: int, height: int, cam_id: int) -> None:
    if int(box.image_width) != width or int(box.image_height) != height:
        raise ValueError(
            f"BoxTracks image dimensions for camera {cam_id}, frame {box.frame_index} are "
            f"{box.image_width}x{box.image_height}, but the supplied video is {width}x{height}. "
            "Use the source video that produced these boxes."
        )


def _unrotate_keypoints(xy: np.ndarray, rotation_count: int, width: int, height: int) -> np.ndarray:
    """Return rotated-frame pixel centres in source coordinates.

    ``unrotate_points`` models image edges, which is appropriate for boxes.
    Keypoints are pixel centres, so their valid coordinate extents end at
    width - 1 and height - 1.  Keep that distinction local to this new path
    rather than changing legacy tracker rotation behaviour.
    """
    return unrotate_points(xy, rotation_count, width - 1, height - 1)


def _empty_image_points() -> ImagePoints:
    return ImagePoints(
        pd.DataFrame(
            columns=[
                "sync_index",
                "cam_id",
                "frame_index",
                "frame_time",
                "object_id",
                "keypoint_id",
                "img_loc_x",
                "img_loc_y",
                "obj_loc_x",
                "obj_loc_y",
                "obj_loc_z",
                "confidence",
                "track_id",
            ]
        )
    )


def _rotate_box(
    box: tuple[float, float, float, float], rotation_count: int, width: int, height: int
) -> tuple[float, float, float, float]:
    points = np.array([[box[0], box[1]], [box[0], box[3]], [box[2], box[1]], [box[2], box[3]]], dtype=float)
    if rotation_count in (1, -3):
        rotated = np.column_stack((height - points[:, 1], points[:, 0]))
    elif rotation_count in (2, -2):
        rotated = np.column_stack((width - points[:, 0], height - points[:, 1]))
    elif rotation_count in (-1, 3):
        rotated = np.column_stack((points[:, 1], width - points[:, 0]))
    else:
        rotated = points
    return (
        float(rotated[:, 0].min()),
        float(rotated[:, 1].min()),
        float(rotated[:, 0].max()),
        float(rotated[:, 1].max()),
    )


def _unrotate_box(box: np.ndarray, rotation_count: int, width: int, height: int) -> tuple[float, float, float, float]:
    box_tuple: tuple[float, float, float, float] = (
        float(box[0]),
        float(box[1]),
        float(box[2]),
        float(box[3]),
    )
    if rotation_count % 2:
        return _rotate_box(box_tuple, -rotation_count, height, width)
    return _rotate_box(box_tuple, -rotation_count, width, height)
