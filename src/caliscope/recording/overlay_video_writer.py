"""Write annotated overlay videos from a synchronized-processing run.

OverlayVideoWriter is a passive sink for process_synchronized_recording's
on_frame_data callback. It draws each camera's tracked points onto its frame and
encodes one H.264 file per camera via PyAV. No threads, no synchronizer, no CSV --
it only draws and encodes.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import av
from av.container import OutputContainer
from av.video.stream import VideoStream
from numpy.typing import NDArray

from caliscope.recording.overlay import draw_scatter_overlay
from caliscope.tracker import Tracker

if TYPE_CHECKING:
    from caliscope.core.process_synchronized_recording import FrameData

logger = logging.getLogger(__name__)


class OverlayVideoWriter:
    """Draws tracked-point overlays and encodes one H.264 file per camera.

    Driven by process_synchronized_recording's on_frame_data callback. A per-camera
    PyAV writer is opened lazily on that camera's first frame (dimensions taken from
    the frame). close() must run in the worker's finally so a cancelled or failed run
    still flushes the encoders and leaves decodable files.
    """

    def __init__(self, destination_folder: Path, tracker: Tracker, fps: float, suffix: str | None = None):
        self._destination_folder = destination_folder
        self._tracker = tracker
        self._rate = max(1, round(fps))  # PyAV wants a positive integer rate
        self._suffix = f"_{suffix}" if suffix else ""
        self._writers: dict[int, tuple[OutputContainer, VideoStream]] = {}
        self._closed = False

    def on_frame_data(self, sync_index: int, frame_data: "dict[int, FrameData]") -> None:
        for cam_id, fd in frame_data.items():
            drawn = draw_scatter_overlay(
                fd.frame, fd.points, self._tracker.scatter_draw_instructions, self._tracker.pixel_format
            )
            drawn = _crop_to_even(drawn)  # yuv420p / libx264 require even dimensions
            _, stream = self._writer_for(cam_id, drawn)
            video_frame = av.VideoFrame.from_ndarray(drawn, format="bgr24")
            for packet in stream.encode(video_frame):
                self._writers[cam_id][0].mux(packet)

    def close(self) -> None:
        """Flush each encoder and close every container. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for cam_id, (container, stream) in self._writers.items():
            try:
                for packet in stream.encode():  # flush buffered frames
                    container.mux(packet)
            finally:
                container.close()
        self._writers.clear()

    def _writer_for(self, cam_id: int, frame: NDArray[Any]) -> "tuple[OutputContainer, VideoStream]":
        """Return this camera's writer, opening it lazily on the first frame."""
        if cam_id not in self._writers:
            path = self._destination_folder / f"cam_{cam_id}{self._suffix}.mp4"
            height, width = frame.shape[:2]
            logger.info(f"Opening overlay writer for cam {cam_id} at {path} ({width}x{height} @ {self._rate}fps)")
            container = av.open(str(path), mode="w")
            stream = container.add_stream("h264", rate=self._rate)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            self._writers[cam_id] = (container, stream)
        return self._writers[cam_id]


def _crop_to_even(frame: NDArray[Any]) -> NDArray[Any]:
    """Trim to even width/height; yuv420p subsampling rejects odd dimensions."""
    height, width = frame.shape[:2]
    return frame[: height - (height % 2), : width - (width % 2)]
