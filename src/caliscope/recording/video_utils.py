"""Video file utilities.

Functions for reading video metadata without full frame decoding.
"""

import logging
from pathlib import Path
from typing import TypedDict

import cv2

logger = logging.getLogger(__name__)


class VideoProperties(TypedDict):
    """Video metadata returned by read_video_properties."""

    fps: float
    frame_count: int
    width: int
    height: int
    size: tuple[int, int]


def read_video_properties(source_path: Path) -> VideoProperties:
    """Read video metadata (fps, frame_count, dimensions).

    Opens the video file briefly to extract metadata, then releases the handle.

    Args:
        source_path: Path to video file (.mp4, .avi, etc.)

    Returns:
        VideoProperties dict with keys: fps, frame_count, width, height, size

    Raises:
        ValueError: If video file cannot be opened
    """
    logger.info(f"Reading video properties from: {source_path}")

    video = cv2.VideoCapture(str(source_path))

    if not video.isOpened():
        raise ValueError(f"Could not open the video file: {source_path}")

    try:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return VideoProperties(
            fps=video.get(cv2.CAP_PROP_FPS),
            frame_count=int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=width,
            height=height,
            size=(width, height),
        )
    finally:
        video.release()
