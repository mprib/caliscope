"""Caliscope scripting API -- multicamera calibration without the GUI.

Quick start (pre-calibrated cameras):

    from caliscope.api import CameraArray, CaptureVolume, CharucoTracker, Charuco, extract_image_points

    charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=3.0)
    cameras = CameraArray.from_toml("camera_array.toml")

    videos = {0: "extrinsic/cam_0.mp4", 1: "extrinsic/cam_1.mp4"}
    points = extract_image_points(videos, CharucoTracker(charuco))
    volume = CaptureVolume.bootstrap(points, cameras).optimize()
    volume.save("capture_volume")

All spatial coordinates are in meters when using Charuco-based calibration.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

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
    "calibrate_intrinsics",
    # Exceptions
    "CalibrationError",
]


def extract_image_points(
    videos: Mapping[int, Path | str],
    tracker: Tracker,
    *,
    frame_step: int = 1,
    progress: ProgressCallback | None = None,
) -> ImagePoints:
    """Extract 2D landmark observations from video files.

    Opens each video with PyAV, runs the tracker frame-by-frame, and assembles
    results into a validated ImagePoints DataFrame.

    Args:
        videos: Mapping of camera ID to video file path.
        tracker: Tracker instance to apply to each frame.
        frame_step: Process every Nth frame (default 1 = every frame).
            For intrinsic calibration, frame_step=5 is typical since
            only ~30 diverse frames are needed. For extrinsic calibration,
            higher density may improve triangulation quality.
        progress: Optional callback invoked per-frame for progress reporting.

    Note:
        For multi-camera use (extrinsic calibration), videos must be
        synchronized: frame N in each video must correspond to the same
        moment in time. Single-camera use (intrinsic calibration) has
        no synchronization requirement.

    Raises:
        CalibrationError: If no points detected across all videos.
        FileNotFoundError: If any video paths do not exist.
        ValueError: If frame_step < 1.
    """
    import av
    import pandas as pd
    from caliscope.recording.video_utils import read_video_properties

    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")

    # Validate all paths upfront
    missing = {cam_id: str(Path(p)) for cam_id, p in videos.items() if not Path(p).exists()}
    if missing:
        detail = "\n".join(f"  cam {cid}: {p}" for cid, p in missing.items())
        raise FileNotFoundError(f"Video files not found:\n{detail}")

    all_rows: list[dict] = []

    for cam_id, video_path in videos.items():
        video_path = Path(video_path)
        rotation_count = 0

        container = av.open(str(video_path))
        video_stream = container.streams.video[0]
        time_base = float(video_stream.time_base) if video_stream.time_base is not None else 0.0

        props = read_video_properties(video_path)
        frame_count = props["frame_count"]
        # Progress total reflects frames that will actually be processed
        progress_total = (frame_count + frame_step - 1) // frame_step

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
