"""Independent ffmpeg frame oracle for video-I/O tests.

Decodes a specific frame by absolute presentation-order index using the ffmpeg
CLI, giving a ground truth that does NOT share PyAV's decode/seek code path.

Invariants that keep ffmpeg's frame count `n` aligned with PyAV's decode-order
enumerate index:
  - No `-ss` (input seek would reset/offset the `n` counter).
  - `-vsync 0` / passthrough so swscale/vsync cannot duplicate or drop a frame.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _dims(video_path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    w, h = out.stdout.strip().split("x")
    return int(w), int(h)


def dump_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """Return frame `frame_index` (decode/presentation order) as a BGR ndarray."""
    w, h = _dims(video_path)
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(out.stdout, dtype=np.uint8).reshape(h, w, 3)
