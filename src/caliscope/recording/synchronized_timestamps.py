"""Multi-camera temporal alignment for synchronized recordings.

SynchronizedTimestamps holds per-camera frame timestamps and derives the
sync mapping (which frames across cameras correspond to the same moment
in time).
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from statistics import mean
from types import MappingProxyType
from typing import Self

import pandas as pd

from caliscope.recording.frame_timestamps import FrameTimestamps
from caliscope.recording.video_utils import read_video_properties

logger = logging.getLogger(__name__)

# Internal type alias for the sync mapping
_SyncMapping = dict[int, dict[int, int | None]]


@dataclass(frozen=True)
class SynchronizedTimestamps:
    """Temporal alignment of a multi-camera recording.

    Holds per-camera frame timestamps and derives the sync mapping
    (which frames across cameras correspond to the same moment in time).

    Constructed via factory methods, not directly. The sync mapping is
    computed once and cached internally. It is never exposed -- consumers
    use frame_for() and time_for() instead.

    Usage:
        synced = SynchronizedTimestamps.load(recording_dir, cam_ids)
        for sync_index in synced.sync_indices:
            for cam_id in synced.cam_ids:
                frame_index = synced.frame_for(sync_index, cam_id)
                if frame_index is not None:
                    frame_time = synced.time_for(cam_id, frame_index)
    """

    _camera_timestamps: Mapping[int, FrameTimestamps]

    # -------------------------------------------------------------------------
    # Query methods (public API)
    # -------------------------------------------------------------------------

    @cached_property
    def sync_indices(self) -> list[int]:
        """Sorted list of valid sync indices."""
        return sorted(self._cache.keys())

    @property
    def cam_ids(self) -> list[int]:
        """Camera IDs in this recording, sorted."""
        return sorted(self._camera_timestamps.keys())

    def frame_for(self, sync_index: int, cam_id: int) -> int | None:
        """Frame index for a camera at a sync index. None if dropped."""
        return self._cache[sync_index][cam_id]

    def time_for(self, cam_id: int, frame_index: int) -> float:
        """Wall-clock timestamp for a camera's frame."""
        return self._camera_timestamps[cam_id].frame_times[frame_index]

    def for_camera(self, cam_id: int) -> FrameTimestamps:
        """Per-camera timestamps (for streaming path consumers)."""
        return self._camera_timestamps[cam_id]

    def to_csv(self, path: Path) -> None:
        """Write all camera timestamps to CSV (cam_id, frame_time format)."""
        rows: list[dict] = []
        for cam_id in self.cam_ids:
            ft = self._camera_timestamps[cam_id]
            for frame_index in sorted(ft.frame_times.keys()):
                rows.append({"cam_id": cam_id, "frame_time": ft.frame_times[frame_index]})
        pd.DataFrame(rows).to_csv(path, index=False)

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    @cached_property
    def _cache(self) -> _SyncMapping:
        """Sync mapping computed from camera timestamps. Private."""
        return self._compute_sync_mapping()

    def _compute_sync_mapping(self) -> _SyncMapping:
        """Greedy forward-pass sync algorithm.

        Reads from _camera_timestamps and computes which frame from each camera
        belongs to each sync group, handling dropped frames and slight
        timing differences.
        """
        frames_by_cam: dict[int, list[float]] = {
            cam_id: [ft.frame_times[i] for i in sorted(ft.frame_times.keys())]
            for cam_id, ft in self._camera_timestamps.items()
        }

        cam_ids = sorted(frames_by_cam.keys())
        cursors = {cam_id: 0 for cam_id in cam_ids}

        sync_map: _SyncMapping = {}
        sync_index = 0

        while any(cursors[c] < len(frames_by_cam[c]) for c in cam_ids):
            candidates: dict[int, float] = {}
            for cam_id in cam_ids:
                if cursors[cam_id] < len(frames_by_cam[cam_id]):
                    candidates[cam_id] = frames_by_cam[cam_id][cursors[cam_id]]

            if not candidates:
                break

            earliest_next: dict[int, float] = {}
            latest_current: dict[int, float] = {}

            for cam_id in cam_ids:
                earliest_next[cam_id] = _earliest_next_frame(cam_id, cursors, frames_by_cam)
                latest_current[cam_id] = _latest_current_frame(cam_id, cursors, frames_by_cam)

            assigned: dict[int, int | None] = {}
            for cam_id in cam_ids:
                if cam_id not in candidates:
                    assigned[cam_id] = None
                    continue

                frame_time = candidates[cam_id]
                current_frame_index = cursors[cam_id]

                if frame_time > earliest_next[cam_id]:
                    assigned[cam_id] = None
                    continue

                delta_to_next = earliest_next[cam_id] - frame_time
                delta_to_current = frame_time - latest_current[cam_id]

                if delta_to_next < delta_to_current:
                    assigned[cam_id] = None
                    continue

                assigned[cam_id] = current_frame_index
                cursors[cam_id] += 1

            if any(v is not None for v in assigned.values()):
                sync_map[sync_index] = assigned
                sync_index += 1
            else:
                if candidates:
                    min_cam = min(candidates.keys(), key=lambda c: candidates[c])
                    cursors[min_cam] += 1

        return sync_map

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_csv(cls, recording_dir: Path) -> Self:
        """Convenience wrapper: loads from timestamps.csv in recording_dir.

        Delegates to from_csv_path() with the canonical path
        ``recording_dir / "timestamps.csv"``.
        """
        csv_path = recording_dir / "timestamps.csv"
        return cls.from_csv_path(csv_path)

    @classmethod
    def from_csv_path(cls, csv_path: Path) -> Self:
        """Load from an explicit timestamps CSV path.

        Reads the CSV, groups rows by cam_id, and constructs a FrameTimestamps
        per camera. The sync_index column is ignored if present -- the sync
        mapping is always recomputed from timestamps. Use this when the CSV
        lives outside a canonical recording directory (e.g., scripting API
        with user-supplied paths).
        """
        df = pd.read_csv(csv_path)

        camera_timestamps: dict[int, FrameTimestamps] = {}
        for cam_key, group in df.groupby("cam_id"):
            cam_id = int(cam_key)  # type: ignore[arg-type]  # pandas groupby key is numpy.int64
            sorted_group = group.sort_values("frame_time").reset_index(drop=True)
            frame_times = {i: float(t) for i, t in enumerate(sorted_group["frame_time"])}
            camera_timestamps[cam_id] = FrameTimestamps(MappingProxyType(frame_times))

        logger.debug(f"Loaded timestamps from CSV for {len(camera_timestamps)} cameras")
        return cls(MappingProxyType(camera_timestamps))

    @classmethod
    def from_video_paths(cls, videos: Mapping[int, Path]) -> Self:
        """Infer timestamps from explicit video file paths.

        Accepts a cam_id -> video path mapping directly, rather than constructing
        paths from a recording directory. Does NOT write inferred_timestamps.csv
        (no canonical directory to write into).

        from_videos() is for the GUI's directory-based workflow;
        from_video_paths() is for the scripting API where callers pass explicit paths.

        Raises:
            ValueError: If mapping is empty, frame count is 0, or FPS is missing.
            FileNotFoundError: If any video path does not exist.
        """
        if not videos:
            raise ValueError("No video paths provided -- cannot infer timestamps")

        props_by_cam: dict[int, tuple[int, float]] = {}  # cam_id -> (frame_count, fps)

        for cam_id, video_path in videos.items():
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            props = read_video_properties(video_path)
            frame_count = props["frame_count"]
            fps = props["fps"]

            if frame_count <= 0:
                raise ValueError(f"Could not determine frame count for cam_{cam_id}")

            if fps <= 0:
                raise ValueError(
                    f"Could not determine frame rate for cam_{cam_id}. "
                    f"The video file may be corrupted or in an unsupported format."
                )

            props_by_cam[cam_id] = (frame_count, fps)

        # Warn on FPS differences
        all_fps = [fps for _, fps in props_by_cam.values()]
        max_fps = max(all_fps)
        min_fps = min(all_fps)
        if len(all_fps) > 1 and (max_fps - min_fps) / max_fps >= 0.01:
            fps_summary = ", ".join(f"cam_{c}: {f:.2f} fps" for c, (_, f) in sorted(props_by_cam.items()))
            logger.warning(
                f"Frame rates differ across cameras ({fps_summary}). "
                f"Inferred timestamps may not reflect actual recording timing. "
                f"For best results, provide a timestamps.csv with per-frame timing."
            )

        # Warn on frame count differences (inference assumes aligned start)
        counts = {cam_id: fc for cam_id, (fc, _) in props_by_cam.items()}
        max_count = max(counts.values())
        min_count = min(counts.values())

        if max_count != min_count:
            summary = ", ".join(f"cam_{c}: {fc} frames" for c, fc in sorted(counts.items()))
            logger.warning(
                f"Frame counts differ across cameras ({summary}). "
                f"Cameras with fewer frames will have periodic unmatched sync indices "
                f"spread throughout the recording. For best results, provide a "
                f"timestamps.csv with per-frame timing."
            )

        # Compute shared average duration (Amendment 10: each camera uses its own count)
        avg_duration = mean(fc / fps for fc, fps in props_by_cam.values())

        camera_timestamps: dict[int, FrameTimestamps] = {}
        for cam_id, (frame_count, _) in props_by_cam.items():
            frame_times = {i: i * avg_duration / frame_count for i in range(frame_count)}
            camera_timestamps[cam_id] = FrameTimestamps(MappingProxyType(frame_times))

        total_frames = sum(fc for fc, _ in props_by_cam.values())
        logger.info(
            f"Inferred timestamps for {len(props_by_cam)} cameras, "
            f"{total_frames} total frames, avg duration {avg_duration:.3f}s"
        )

        return cls(MappingProxyType(camera_timestamps))

    @classmethod
    def from_videos(cls, recording_dir: Path, cam_ids: Sequence[int]) -> Self:
        """Infer timestamps from video metadata.

        Requires cam_ids because there is no CSV to discover them from.
        Validates that frame counts and FPS are compatible across cameras.
        Persists the result as inferred_timestamps.csv (write-only audit trail,
        never read back as input).

        from_videos() is for the GUI's directory-based workflow;
        from_video_paths() is for the scripting API where callers pass explicit paths.

        Raises:
            ValueError: If no cam_ids given, no video found, or frame count is 0.
        """
        if not cam_ids:
            raise ValueError("No cam_ids provided -- cannot infer timestamps")

        video_paths: dict[int, Path] = {}
        for cam_id in cam_ids:
            video_path = recording_dir / f"cam_{cam_id}.mp4"
            if not video_path.exists():
                raise ValueError(f"Video file not found: {video_path}")
            video_paths[cam_id] = video_path

        result = cls.from_video_paths(video_paths)

        # Persist as write-only audit trail (only for directory-based workflow)
        inferred_path = recording_dir / "inferred_timestamps.csv"
        result.to_csv(inferred_path)

        return result

    @classmethod
    def load(cls, recording_dir: Path, cam_ids: Sequence[int]) -> Self:
        """Smart factory: uses timestamps.csv if available, else infers from videos.

        This is what the presenter calls. Raises ValueError if inference fails
        (e.g., incompatible frame counts or FPS).
        """
        csv_path = recording_dir / "timestamps.csv"
        if csv_path.exists():
            logger.debug(f"Loading timestamps from CSV: {csv_path}")
            return cls.from_csv(recording_dir)

        logger.info(f"No timestamps.csv found in {recording_dir}, inferring from videos")
        return cls.from_videos(recording_dir, cam_ids)


# -------------------------------------------------------------------------
# Module-level helpers (used by _compute_sync_mapping)
# -------------------------------------------------------------------------


def _earliest_next_frame(cam_id: int, cursors: dict[int, int], frames_by_cam: dict[int, list[float]]) -> float:
    """Get minimum frame_time of NEXT frames from OTHER cameras."""
    times = []
    for c in cursors:
        if c == cam_id:
            continue
        next_index = cursors[c] + 1
        if next_index < len(frames_by_cam[c]):
            times.append(frames_by_cam[c][next_index])
    return min(times) if times else float("inf")


def _latest_current_frame(cam_id: int, cursors: dict[int, int], frames_by_cam: dict[int, list[float]]) -> float:
    """Get maximum frame_time of CURRENT frames from OTHER cameras."""
    times = []
    for c in cursors:
        if c == cam_id:
            continue
        current_index = cursors[c]
        if current_index < len(frames_by_cam[c]):
            times.append(frames_by_cam[c][current_index])
    return max(times) if times else float("-inf")
