"""Video file utilities.

Functions for reading video metadata without full frame decoding.
Uses PyAV for container inspection -- no frames are decoded.
"""

import logging
from pathlib import Path
from typing import TypedDict

import av

logger = logging.getLogger(__name__)


class VideoProperties(TypedDict):
    """Video metadata returned by read_video_properties."""

    fps: float
    frame_count: int
    width: int
    height: int
    size: tuple[int, int]


def read_video_properties(source_path: Path) -> VideoProperties:
    """Read video metadata (fps, frame_count, dimensions) via PyAV.

    Opens the video container briefly to inspect stream metadata,
    then closes it. No frames are decoded.

    Falls back to duration-based frame count when the container
    does not report stream.frames (same logic as FrameSource.frame_count).

    Raises:
        FileNotFoundError: If source_path does not exist.
        ValueError: If the file cannot be opened as video, has no video
            stream, or if fps/frame_count cannot be determined. All PyAV
            exceptions are wrapped as ValueError to preserve the caller
            contract.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_path}")

    logger.info(f"Reading video properties from: {source_path}")

    try:
        container = av.open(str(source_path))
    except Exception as e:
        raise ValueError(
            f"Could not open video file: {source_path}. The file may be corrupted or in an unsupported format."
        ) from e

    try:
        if not container.streams.video:
            raise ValueError(f"No video stream found in: {source_path}")

        stream = container.streams.video[0]

        # FPS from average_rate (rational number, e.g. 30000/1001)
        if stream.average_rate is None or float(stream.average_rate) <= 0:
            raise ValueError(
                f"Could not determine frame rate for: {source_path}. "
                f"The video file may be corrupted or in an unsupported format."
            )
        fps = float(stream.average_rate)

        # Frame count: prefer stream metadata, fall back to duration * fps
        # This matches FrameSource.frame_count (metadata estimate), not
        # FrameSource.last_frame_index (keyframe-scan result).
        frame_count = stream.frames
        if frame_count == 0 and container.duration is not None:
            duration_seconds = container.duration / 1_000_000
            frame_count = int(duration_seconds * fps)

        if frame_count <= 0:
            raise ValueError(
                f"Could not determine frame count for: {source_path}. "
                f"stream.frames={stream.frames}, "
                f"container.duration={container.duration}"
            )

        width = stream.width
        height = stream.height

        return VideoProperties(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            size=(width, height),
        )
    finally:
        container.close()
