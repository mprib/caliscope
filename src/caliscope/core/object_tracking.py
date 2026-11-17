"""Box tracking and box-limited keypoint and embedding passes for synchronized recordings.

This module keeps detection, camera-local box tracking, keypoint estimation, and
re-identification embedding extraction as separate passes.  Boxes are stored in
the original video coordinate system so they can be inspected and persisted
without depending on inference rotation.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from caliscope.core.box_tracks import BOX_TRACK_COLUMNS, BoxTracks
from caliscope.core.point_data import ImagePoints
from caliscope.core.reidentification_embeddings import KEY_COLUMNS, ReidentificationEmbeddings
from caliscope.recording.frame_source import FrameSource
from caliscope.reporting import ProgressCallback
from caliscope.trackers.byte_tracker import ByteTracker, LOW_SCORE_THRESHOLD
from caliscope.trackers.helper import apply_rotation, unrotate_points

if TYPE_CHECKING:
    from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps


class ObjectDetector(Protocol):
    def detect(self, frame: np.ndarray, min_score: float | None = None) -> tuple[np.ndarray, np.ndarray]: ...


class BoxKeypointTracker(Protocol):
    pixel_format: Any

    def keypoints_in_box(
        self, frame: np.ndarray, box: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, np.ndarray]: ...


class CropEncoder(Protocol):
    def encode(self, crops: Sequence[np.ndarray]) -> np.ndarray: ...


# Crops narrower or shorter than this many pixels after clamping to the frame are skipped.
MIN_CROP_SIDE_PX = 2


def track_objects(
    videos: Mapping[int, Path | str],
    detector: ObjectDetector,
    *,
    timestamps: SynchronizedTimestamps,
    frame_step: int = 1,
    rotation_counts: Mapping[int, int] | None = None,
    progress: ProgressCallback | None = None,
) -> BoxTracks:
    """Detect and track every object box at sampled synchronized moments.

    ``timestamps`` sets the alignment policy. ``frame_step`` counts synchronized
    moments, and a camera with no frame at a sampled moment is skipped there.

    Every returned row is unlinked: object_id is negative and unique per camera
    and track. IDs run -1, -2, ... over the (cam_id, track_id) pairs in
    ascending order.
    """
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")
    video_paths = _video_paths(videos)
    missing = set(video_paths) - set(timestamps.cam_ids)
    if missing:
        raise ValueError(f"Timestamps have no entry for cameras {sorted(missing)}")
    selected_sync_indices = timestamps.sync_indices[::frame_step]
    rotations = rotation_counts or {}
    decode_threads = max(1, (os.cpu_count() or 4) // len(video_paths))

    def process_camera(cam_id: int, path: Path) -> list[dict[str, Any]]:
        work = _sync_work(timestamps, cam_id, selected_sync_indices)
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
                boxes, scores = detector.detect(inference_frame, min_score=LOW_SCORE_THRESHOLD)
                for track_id, detected_box, detector_score in tracker.update(boxes, scores):
                    source_box = _unrotate_box(detected_box, rotation_count, raw.frame.shape[1], raw.frame.shape[0])
                    rows.append(
                        {
                            "sync_index": sync_for_frame[raw.frame_index],
                            "cam_id": cam_id,
                            "frame_index": raw.frame_index,
                            "frame_time": timestamps.time_for(cam_id, raw.frame_index),
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

    rows = _run_per_camera(video_paths, process_camera)
    # Assigned after all workers finish so thread completion order cannot change the IDs.
    camera_tracks = sorted({(int(row["cam_id"]), int(row["track_id"])) for row in rows})
    unlinked_ids = {camera_track: -(position + 1) for position, camera_track in enumerate(camera_tracks)}
    for row in rows:
        row["object_id"] = unlinked_ids[(int(row["cam_id"]), int(row["track_id"]))]
    return BoxTracks(pd.DataFrame(rows, columns=list(BOX_TRACK_COLUMNS)))


def estimate_keypoints(
    videos: Mapping[int, Path | str],
    keypoint_tracker: BoxKeypointTracker,
    box_tracks: BoxTracks,
    *,
    min_confidence: float,
    progress: ProgressCallback | None = None,
) -> ImagePoints:
    """Estimate keypoints in each supplied box, copying object_id and track_id from it.

    Accepts several boxes per camera and frame, one per object (guaranteed by
    BoxTracks validation). Serves both the sampled association pass (unlinked
    boxes, negative object IDs) and the full-rate pass (linked boxes).

    The recorded rows are the timing authority for this pass.  It does not
    resynchronize videos or infer observations for gaps in the box table.
    """
    # Some supported model decoders expose unnormalised confidence scores, so
    # a hard upper bound would reject otherwise valid tracker output.
    if not np.isfinite(min_confidence) or min_confidence < 0:
        raise ValueError(f"min_confidence must be finite and non-negative, got {min_confidence}")
    boxes = box_tracks.df
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
        by_frame = _rows_by_frame(camera_boxes)
        if progress is not None:
            progress.on_video_start(cam_id, len(by_frame))
        source = FrameSource.from_path(
            path,
            cam_id=cam_id,
            decode_threads=decode_threads,
            wanted_indices=set(by_frame),
            pixel_format=keypoint_tracker.pixel_format,
        )
        rows: list[dict[str, Any]] = []
        seen_frames: set[int] = set()
        try:
            processed = 0
            while (raw := source.next_frame()) is not None:
                seen_frames.add(raw.frame_index)
                kept_in_frame = 0
                for box in by_frame[raw.frame_index]:
                    rotation_count = int(box.rotation_count)
                    _validate_source_geometry(box, raw.frame.shape[1], raw.frame.shape[0], cam_id)
                    inference_frame = apply_rotation(raw.frame, rotation_count)
                    inference_box = _rotate_box(
                        (box.box_x1, box.box_y1, box.box_x2, box.box_y2),
                        rotation_count,
                        raw.frame.shape[1],
                        raw.frame.shape[0],
                    )
                    xy, confidence = keypoint_tracker.keypoints_in_box(inference_frame, inference_box)
                    xy = _unrotate_keypoints(xy, rotation_count, inference_frame.shape[1], inference_frame.shape[0])
                    keep = _valid_keypoints(xy, confidence, min_confidence, raw.frame.shape[1], raw.frame.shape[0])
                    kept_in_frame += int(keep.sum())
                    for keypoint_id in np.flatnonzero(keep):
                        rows.append(
                            {
                                "sync_index": int(box.sync_index),
                                "cam_id": cam_id,
                                "frame_index": int(box.frame_index),
                                "frame_time": float(box.frame_time),
                                "object_id": int(box.object_id),
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
                    progress.on_frame(cam_id, processed, kept_in_frame)
        finally:
            source.close()
        _check_frames_seen(by_frame, seen_frames, cam_id)
        if progress is not None:
            progress.on_video_complete(cam_id)
        return rows

    rows = _run_per_camera({cam_id: video_paths[cam_id] for cam_id in active_cameras}, process_camera)
    return ImagePoints(pd.DataFrame(rows)) if rows else _empty_image_points()


def extract_reidentification_embeddings(
    videos: Mapping[int, Path | str],
    encoder: CropEncoder,
    box_tracks: BoxTracks,
    *,
    progress: ProgressCallback | None = None,
) -> ReidentificationEmbeddings:
    """Encode the crop inside each supplied box as one re-identification embedding.

    Returns one unit-length vector per box keyed by its sync_index, cam_id and
    track_id. Boxes too small to crop after clamping to the frame are skipped.
    It is normally run on the same sampled boxes as the association keypoint pass.

    The recorded rows are the timing authority for this pass.  It does not
    resynchronize videos or infer observations for gaps in the box table.
    """
    boxes = box_tracks.df
    table_cameras = set(boxes["cam_id"].unique())
    missing_cameras = sorted(table_cameras - set(videos))
    if missing_cameras:
        raise ValueError(f"BoxTracks contains camera IDs absent from videos: {missing_cameras}")
    video_paths = _video_paths(videos)

    active_cameras = [cam_id for cam_id in video_paths if cam_id in table_cameras]
    if not active_cameras:
        return _empty_embeddings()
    decode_threads = max(1, (os.cpu_count() or 4) // len(active_cameras))

    def process_camera(cam_id: int, path: Path) -> list[dict[str, Any]]:
        camera_boxes = boxes.loc[boxes["cam_id"] == cam_id].copy()
        by_frame = _rows_by_frame(camera_boxes)
        if progress is not None:
            progress.on_video_start(cam_id, len(by_frame))
        source = FrameSource.from_path(path, cam_id=cam_id, decode_threads=decode_threads, wanted_indices=set(by_frame))
        rows: list[dict[str, Any]] = []
        seen_frames: set[int] = set()
        try:
            processed = 0
            while (raw := source.next_frame()) is not None:
                seen_frames.add(raw.frame_index)
                inference_frames: dict[int, np.ndarray] = {}
                crops: list[np.ndarray] = []
                cropped_boxes: list[Any] = []
                for box in by_frame[raw.frame_index]:
                    rotation_count = int(box.rotation_count)
                    _validate_source_geometry(box, raw.frame.shape[1], raw.frame.shape[0], cam_id)
                    if rotation_count not in inference_frames:
                        inference_frames[rotation_count] = apply_rotation(raw.frame, rotation_count)
                    inference_frame = inference_frames[rotation_count]
                    inference_box = _rotate_box(
                        (box.box_x1, box.box_y1, box.box_x2, box.box_y2),
                        rotation_count,
                        raw.frame.shape[1],
                        raw.frame.shape[0],
                    )
                    crop = _crop_box(inference_frame, inference_box)
                    if crop is not None:
                        crops.append(crop)
                        cropped_boxes.append(box)
                if crops:
                    vectors = np.asarray(encoder.encode(crops), dtype=np.float32)
                    for box, vector in zip(cropped_boxes, vectors, strict=True):
                        rows.append(
                            {
                                "sync_index": int(box.sync_index),
                                "cam_id": cam_id,
                                "track_id": int(box.track_id),
                                "vector": vector,
                            }
                        )
                processed += 1
                if progress is not None:
                    progress.on_frame(cam_id, processed, len(crops))
        finally:
            source.close()
        _check_frames_seen(by_frame, seen_frames, cam_id)
        if progress is not None:
            progress.on_video_complete(cam_id)
        return rows

    rows = _run_per_camera({cam_id: video_paths[cam_id] for cam_id in active_cameras}, process_camera)
    if not rows:
        return _empty_embeddings()
    index = pd.DataFrame([{column: row[column] for column in KEY_COLUMNS} for row in rows], columns=KEY_COLUMNS)
    return ReidentificationEmbeddings(index, np.stack([row["vector"] for row in rows]))


def _crop_box(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """Cut a box out of a frame, widened to whole pixels and clamped to the frame."""
    height, width = frame.shape[:2]
    x1 = min(max(math.floor(box[0]), 0), width)
    y1 = min(max(math.floor(box[1]), 0), height)
    x2 = min(max(math.ceil(box[2]), 0), width)
    y2 = min(max(math.ceil(box[3]), 0), height)
    if x2 - x1 < MIN_CROP_SIDE_PX or y2 - y1 < MIN_CROP_SIDE_PX:
        return None
    return frame[y1:y2, x1:x2]


def _empty_embeddings() -> ReidentificationEmbeddings:
    return ReidentificationEmbeddings(pd.DataFrame(columns=KEY_COLUMNS), np.empty((0, 0), dtype=np.float32))


def _check_frames_seen(by_frame: Mapping[int, Any], seen_frames: set[int], cam_id: int) -> None:
    missing_frames = sorted(set(by_frame) - seen_frames)
    if missing_frames:
        shown = ", ".join(str(frame_index) for frame_index in missing_frames[:10])
        suffix = "..." if len(missing_frames) > 10 else ""
        raise ValueError(
            f"BoxTracks references frames unavailable in video for camera {cam_id}: {shown}{suffix}. "
            "Regenerate boxes from this video or select only available frames."
        )


def _video_paths(videos: Mapping[int, Path | str]) -> dict[int, Path]:
    if not videos:
        raise ValueError("videos must contain at least one camera")
    paths = {int(cam_id): Path(path) for cam_id, path in videos.items()}
    missing = {cam_id: path for cam_id, path in paths.items() if not path.exists()}
    if missing:
        detail = "\n".join(f"  cam {cam_id}: {path}" for cam_id, path in missing.items())
        raise FileNotFoundError(f"Video files not found:\n{detail}")
    return paths


def _sync_work(timestamps: SynchronizedTimestamps, cam_id: int, sync_indices: list[int]) -> list[tuple[int, int]]:
    return [
        (sync_index, frame_index)
        for sync_index in sync_indices
        if (frame_index := timestamps.frame_for(sync_index, cam_id)) is not None
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


def _rows_by_frame(boxes: pd.DataFrame) -> dict[int, list[Any]]:
    by_frame: dict[int, list[Any]] = {}
    for frame_index, row in zip(boxes["frame_index"].tolist(), boxes.itertuples(index=False)):
        by_frame.setdefault(int(frame_index), []).append(row)
    return by_frame


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
