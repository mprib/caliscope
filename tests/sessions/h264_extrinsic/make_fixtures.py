"""Generate synthetic H.264 / B-frame / 59.94fps test fixtures.

These fixtures exist to exercise frame-index selection on the codec family that
matters for the field data (H.264, B-frames, fractional 60000/1001 rate) WITHOUT
using anyone's private footage. The content is ffmpeg lavfi synthetic sources, so
the clips are fully license-free and reproducible.

Each frame is visually distinct (built-in counter + motion), so an argmin-SAD
identity check against an independent ffmpeg dump strictly distinguishes a frame
from its neighbors.

Run from anywhere:  uv run python tests/sessions/h264_extrinsic/make_fixtures.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

RATE = "60000/1001"
N_FRAMES = 180
HERE = Path(__file__).parent
# Distinct synthetic sources per camera so multi-camera tests can tell them apart.
SOURCES = {"cam_0.mp4": "testsrc2", "cam_1.mp4": "testsrc"}


def main() -> None:
    for name, src in SOURCES.items():
        out = HERE / name
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"{src}=size=320x240:rate={RATE}",
                "-frames:v",
                str(N_FRAMES),
                "-c:v",
                "libx264",
                "-bf",
                "2",
                "-g",
                "13",
                "-pix_fmt",
                "yuv420p",
                str(out),
            ],
            check=True,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
