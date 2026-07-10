"""Drive OverlayVideoWriter end to end and confirm it produces decodable files."""

from pathlib import Path

import av
import numpy as np

from caliscope.core.process_synchronized_recording import FrameData
from caliscope.packets import PixelFormat
from caliscope.recording.overlay_video_writer import OverlayVideoWriter


class _StubTracker:
    """Minimal stand-in: OverlayVideoWriter only touches these two members."""

    def __init__(self, pixel_format: PixelFormat = PixelFormat.BGR):
        self.pixel_format = pixel_format

    def scatter_draw_instructions(self, keypoint_id: int) -> dict:
        return {"radius": 5, "color": (0, 0, 255), "thickness": -1}


def _decoded_frame_count(path: Path) -> int:
    container = av.open(str(path))
    try:
        return sum(1 for _ in container.decode(video=0))
    finally:
        container.close()


def _feed(writer: OverlayVideoWriter, cam_id: int, n: int, shape: tuple, pixel_format: PixelFormat):
    for i in range(n):
        frame = np.full(shape, 50, dtype=np.uint8)
        writer.on_frame_data(i, {cam_id: FrameData(frame=frame, points=None, frame_index=i)})


def test_full_run_produces_decodable_file(tmp_path: Path):
    writer = OverlayVideoWriter(tmp_path, _StubTracker(), fps=30, suffix="TEST")  # type: ignore[arg-type]
    _feed(writer, cam_id=0, n=10, shape=(480, 640, 3), pixel_format=PixelFormat.BGR)
    writer.close()

    out = tmp_path / "cam_0_TEST.mp4"
    assert out.exists()
    assert _decoded_frame_count(out) == 10


def test_cancelled_run_still_decodable(tmp_path: Path):
    """A short run (close in the worker finally after a cancel) yields a valid file."""
    writer = OverlayVideoWriter(tmp_path, _StubTracker(), fps=30, suffix="TEST")  # type: ignore[arg-type]
    _feed(writer, cam_id=0, n=3, shape=(480, 640, 3), pixel_format=PixelFormat.BGR)
    writer.close()  # simulates finally after cancellation, mid-stream

    assert _decoded_frame_count(tmp_path / "cam_0_TEST.mp4") == 3


def test_grayscale_frames_encode(tmp_path: Path):
    writer = OverlayVideoWriter(tmp_path, _StubTracker(PixelFormat.GRAY), fps=30, suffix="TEST")  # type: ignore[arg-type]
    _feed(writer, cam_id=2, n=5, shape=(480, 640), pixel_format=PixelFormat.GRAY)
    writer.close()

    assert _decoded_frame_count(tmp_path / "cam_2_TEST.mp4") == 5


def test_close_is_idempotent(tmp_path: Path):
    writer = OverlayVideoWriter(tmp_path, _StubTracker(), fps=30)  # type: ignore[arg-type]
    _feed(writer, cam_id=0, n=2, shape=(480, 640, 3), pixel_format=PixelFormat.BGR)
    writer.close()
    writer.close()  # second call must no-op, not raise


if __name__ == "__main__":
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)
    w = OverlayVideoWriter(debug_dir, _StubTracker(), fps=30, suffix="DEBUG")  # type: ignore[arg-type]
    _feed(w, cam_id=0, n=30, shape=(480, 640, 3), pixel_format=PixelFormat.BGR)
    w.close()
    print(f"wrote {debug_dir / 'cam_0_DEBUG.mp4'}, frames={_decoded_frame_count(debug_dir / 'cam_0_DEBUG.mp4')}")
