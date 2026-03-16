"""Caliscope scripting API -- multicamera calibration without the GUI.

Quick start (pre-calibrated cameras):

    from caliscope.api import (
        CameraArray, CaptureVolume, Charuco, CharucoTracker,
        extract_image_points, extract_image_points_multicam,
    )

    charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=3.0)
    cameras = CameraArray.from_toml("camera_array.toml")

    # Progress bars are shown automatically by default.
    points = extract_image_points("extrinsic/cam_0.mp4", 0, CharucoTracker(charuco))
    # Or for synchronized multi-camera extraction:
    videos = {0: "extrinsic/cam_0.mp4", 1: "extrinsic/cam_1.mp4"}
    points = extract_image_points_multicam(videos, CharucoTracker(charuco))
    volume = CaptureVolume.bootstrap(points, cameras).optimize()
    volume.save("capture_volume")

Pass ``progress=None`` to any extraction function to suppress progress output.

All spatial coordinates are in meters when using Charuco-based calibration.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.calibrate_intrinsics import (
    IntrinsicCalibrationOutput,
    IntrinsicCalibrationReport,
)
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.charuco import Charuco
from caliscope.core.point_data import ImagePoints
from caliscope.exceptions import CalibrationError
from caliscope.tracker import Tracker
from caliscope.trackers.charuco_tracker import CharucoTracker

if TYPE_CHECKING:
    from caliscope.reporting import ProgressCallback

# Sentinel meaning "create a RichProgressBar automatically for this call".
# Users never reference this value directly -- it is a private implementation detail.
_AUTO: Any = object()


@contextmanager
def _auto_progress(progress: Any) -> Generator[ProgressCallback | None, None, None]:
    """Resolve the progress parameter and manage the lifecycle of auto-created bars.

    Yields the resolved callback.  When ``_AUTO`` is passed, a ``RichProgressBar``
    is created, started, and stopped around the body.  When ``None`` or a concrete
    callback is passed, it is yielded as-is with no lifecycle management.
    """
    if progress is _AUTO:
        from caliscope.reporting import RichProgressBar

        bar = RichProgressBar()
        with bar:
            yield bar
    else:
        yield progress


__all__ = [
    # Domain classes
    "Charuco",
    "CharucoTracker",
    "CameraData",
    "CameraArray",
    "ImagePoints",
    "CaptureVolume",
    # Result types
    "IntrinsicCalibrationOutput",
    "IntrinsicCalibrationReport",
    # Functions
    "extract_image_points",
    "extract_image_points_multicam",
    "calibrate_intrinsics",
    # Exceptions
    "CalibrationError",
]


def extract_image_points(
    video_path: Path | str,
    cam_id: int,
    tracker: Tracker,
    *,
    frame_step: int = 1,
    progress: ProgressCallback | None = _AUTO,
) -> ImagePoints:
    """Extract 2D landmark observations from a single camera video.

    Opens the video with PyAV, runs the tracker frame-by-frame, and assembles
    results into a validated ImagePoints DataFrame.

    Args:
        video_path: Path to the video file.
        cam_id: Camera ID to assign to all observations in the result.
        tracker: Tracker instance to apply to each frame.
        frame_step: Process every Nth frame (default 1 = every frame).
            For intrinsic calibration, frame_step=5 is typical since
            only ~30 diverse frames are needed.
        progress: Callback invoked per-frame for progress reporting.
            Defaults to a Rich progress bar.  Pass ``None`` to suppress output.

    Raises:
        CalibrationError: If no points are detected in the video.
        FileNotFoundError: If the video path does not exist.
        ValueError: If frame_step < 1.
    """
    import av
    import pandas as pd
    from caliscope.recording.video_utils import read_video_properties

    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with _auto_progress(progress) as progress:
        all_rows: list[dict] = []

        rotation_count = 0

        container = av.open(str(video_path))
        video_stream = container.streams.video[0]
        time_base = float(video_stream.time_base) if video_stream.time_base is not None else 0.0

        props = read_video_properties(video_path)
        frame_count = props["frame_count"]
        # Progress total reflects frames that will actually be processed
        progress_total = (frame_count + frame_step - 1) // frame_step

        if frame_step > 1 and progress is not None:
            progress.on_info(f"Extracting every {frame_step}th frame ({progress_total} of {frame_count})")

        if progress is not None:
            progress.on_video_start(cam_id, progress_total)

        try:
            progress_index = 0
            for frame_index, frame in enumerate(container.decode(video_stream)):
                if frame_index % frame_step != 0:
                    continue

                bgr = frame.to_ndarray(format="bgr24")
                frame_time = frame.pts * time_base if frame.pts is not None else 0.0

                point_packet = tracker.get_points(bgr, cam_id=cam_id, rotation_count=rotation_count)

                n_points = len(point_packet.point_id)
                if n_points > 0:
                    n = n_points
                    row = {
                        "sync_index": [frame_index] * n,
                        "cam_id": [cam_id] * n,
                        "frame_time": [frame_time] * n,
                        "point_id": point_packet.point_id.tolist(),
                        "img_loc_x": point_packet.img_loc[:, 0].tolist(),
                        "img_loc_y": point_packet.img_loc[:, 1].tolist(),
                        "obj_loc_x": point_packet.obj_loc_list[0],
                        "obj_loc_y": point_packet.obj_loc_list[1],
                        "obj_loc_z": point_packet.obj_loc_list[2],
                    }
                    all_rows.append(row)

                progress_index += 1
                if progress is not None:
                    progress.on_frame(cam_id, progress_index, n_points)
        finally:
            container.close()

        if progress is not None:
            progress.on_video_complete(cam_id)

        if not all_rows:
            raise CalibrationError(
                "No landmarks detected in the video. Check that:\n"
                "  1. The calibration target is visible in the video\n"
                "  2. The correct tracker is being used\n"
                "  3. The video file is not corrupted"
            )

        flat_rows: dict[str, list] = {k: [] for k in all_rows[0].keys()}
        for row in all_rows:
            for k, v in row.items():
                flat_rows[k].extend(v)

        return ImagePoints(pd.DataFrame(flat_rows))


def extract_image_points_multicam(
    videos: Mapping[int, Path | str],
    tracker: Tracker,
    *,
    frame_step: int = 1,
    timestamps: Path | str | None = None,
    progress: ProgressCallback | None = _AUTO,
) -> ImagePoints:
    """Extract synchronized 2D landmark observations from multiple camera videos.

    For each sync index (a common moment in time across all cameras), reads the
    corresponding frame from each camera, runs the tracker, and assembles all
    observations into a flat DataFrame. Designed for extrinsic calibration where
    frame correspondence across cameras is required.

    The tracker must be thread-safe: it will be called concurrently from
    multiple threads (one per camera). Stateless trackers (CharucoTracker,
    ArucoTracker) are safe. ONNX trackers with shared session state may
    require external locking.

    Args:
        videos: Mapping of cam_id to video file path.
        tracker: Tracker instance applied to each frame.
        frame_step: Process every Nth sync index (default 1 = every sync index).
            Operates on sync indices, not raw frame indices. For example,
            frame_step=5 processes sync indices 0, 5, 10, ... regardless of
            which raw frame index each camera uses for that sync index.
        timestamps: Optional path to a timestamps CSV file. The CSV must have
            columns ``cam_id`` and ``frame_time`` (one row per frame per camera).
            If omitted, timestamps are inferred from video metadata (FPS and
            frame count). Providing a CSV is recommended for recordings where
            cameras did not start at exactly the same time.
        progress: Callback invoked per-frame for progress reporting.
            Defaults to a Rich progress bar.  Pass ``None`` to suppress output.
            A single instance is shared across all camera threads (safe because
            ``RichProgressBar`` uses a threading lock internally).

    Returns:
        ImagePoints containing columns: sync_index, cam_id, frame_index,
        frame_time, point_id, img_loc_x, img_loc_y, obj_loc_x, obj_loc_y,
        obj_loc_z. Each row is one detected landmark observation.

    Raises:
        CalibrationError: If no landmarks detected across all videos.
        FileNotFoundError: If any video paths (or the timestamps CSV) do not exist.
        ValueError: If frame_step < 1.
    """
    import concurrent.futures
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    from caliscope.recording.frame_source import FrameSource
    from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps

    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")

    # Normalize all video paths upfront
    video_paths: dict[int, Path] = {cam_id: Path(p) for cam_id, p in videos.items()}

    # Validate all video paths at once
    missing = {cam_id: str(p) for cam_id, p in video_paths.items() if not p.exists()}
    if missing:
        detail = "\n".join(f"  cam {cid}: {p}" for cid, p in missing.items())
        raise FileNotFoundError(f"Video files not found:\n{detail}")

    with _auto_progress(progress) as progress:
        # Build sync mapping
        if timestamps is not None:
            synced = SynchronizedTimestamps.from_csv_path(Path(timestamps))
        else:
            synced = SynchronizedTimestamps.from_video_paths(video_paths)

        # Select sync indices honoring frame_step
        selected_sync_indices = synced.sync_indices[::frame_step]

        if frame_step > 1 and progress is not None:
            progress.on_info(
                f"Extracting every {frame_step}th time-aligned frame "
                f"({len(selected_sync_indices)} of {len(synced.sync_indices)})"
            )

        # Per-camera work list: (sync_index, frame_index) pairs where the camera
        # has a valid (non-dropped) frame for that sync index.
        def _build_work_list(cam_id: int) -> list[tuple[int, int]]:
            work: list[tuple[int, int]] = []
            for sync_index in selected_sync_indices:
                frame_index = synced.frame_for(sync_index, cam_id)
                if frame_index is not None:
                    work.append((sync_index, frame_index))
            return work

        def _process_camera(cam_id: int, work_list: list[tuple[int, int]], video_path: Path) -> list[dict]:
            frame_source = FrameSource.from_path(video_path, cam_id=cam_id)
            try:
                if progress is not None:
                    progress.on_video_start(cam_id, len(work_list))

                rows: list[dict] = []
                for processed_count, (sync_index, frame_index) in enumerate(work_list):
                    frame = frame_source.get_frame(frame_index)
                    if frame is None:
                        if progress is not None:
                            progress.on_frame(cam_id, processed_count, 0)
                        continue

                    frame_time = synced.time_for(cam_id, frame_index)
                    point_packet = tracker.get_points(frame, cam_id=cam_id, rotation_count=0)

                    n_points = len(point_packet.point_id)
                    if n_points > 0:
                        row: dict = {
                            "sync_index": [sync_index] * n_points,
                            "cam_id": [cam_id] * n_points,
                            "frame_index": [frame_index] * n_points,
                            "frame_time": [frame_time] * n_points,
                            "point_id": point_packet.point_id.tolist(),
                            "img_loc_x": point_packet.img_loc[:, 0].tolist(),
                            "img_loc_y": point_packet.img_loc[:, 1].tolist(),
                            "obj_loc_x": point_packet.obj_loc_list[0],
                            "obj_loc_y": point_packet.obj_loc_list[1],
                            "obj_loc_z": point_packet.obj_loc_list[2],
                        }
                        rows.append(row)

                    if progress is not None:
                        progress.on_frame(cam_id, processed_count, n_points)

                if progress is not None:
                    progress.on_video_complete(cam_id)

                return rows
            finally:
                frame_source.close()

        # Process cameras concurrently
        max_workers = min(len(video_paths), 8)
        all_camera_rows: list[list[dict]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_camera, cam_id, _build_work_list(cam_id), video_paths[cam_id]): cam_id
                for cam_id in video_paths
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    all_camera_rows.append(future.result())
            except Exception:
                for f in futures:
                    f.cancel()
                raise

        # Flatten per-camera row lists
        all_rows: list[dict] = [row for camera_rows in all_camera_rows for row in camera_rows]

        if not all_rows:
            raise CalibrationError(
                "No landmarks detected in any video. Check that:\n"
                "  1. The calibration target is visible in the videos\n"
                "  2. The correct tracker is being used\n"
                "  3. Video files are not corrupted"
            )

        flat_rows: dict[str, list] = {k: [] for k in all_rows[0].keys()}
        for row in all_rows:
            for k, v in row.items():
                flat_rows[k].extend(v)

        return ImagePoints(pd.DataFrame(flat_rows))


def calibrate_intrinsics(
    image_points: ImagePoints,
    camera: CameraData,
) -> IntrinsicCalibrationOutput:
    """Calibrate camera intrinsic parameters from 2D observations.

    Args:
        image_points: 2D observations with obj_loc columns (from charuco tracking).
        camera: Camera to calibrate (provides cam_id, image size, fisheye flag).

    Returns:
        IntrinsicCalibrationOutput with .camera and .report.

    Raises:
        CalibrationError: If obj_loc data missing or calibration fails.
    """
    from caliscope.core.calibrate_intrinsics import run_intrinsic_calibration

    # Validate obj_loc presence
    if image_points.df[["obj_loc_x", "obj_loc_y"]].isna().all().all():
        raise CalibrationError(
            "ImagePoints contain no object location data (obj_loc columns are all NaN). "
            "Intrinsic calibration requires a tracker that provides known 3D positions "
            "(e.g., CharucoTracker). Body pose trackers (ONNX) do not provide object locations."
        )

    try:
        return run_intrinsic_calibration(camera, image_points)
    except ValueError as e:
        raise CalibrationError(str(e)) from e
