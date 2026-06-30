"""Forward-only frame access for recorded video files.

FrameSource wraps PyAV for sequential video decoding. One forward pass, no
seeking, no random access. next_frame() returns the next frame as a FramePacket.
When wanted_indices is set at construction,
unwanted frames are decoded but not converted to BGR — next_frame silently
advances past them.

This is infrastructure (I/O), not domain logic — hence placement in recording/
rather than core/.
"""

import logging
from collections.abc import Set
from pathlib import Path
from threading import Lock
from typing import Iterator, Self

import numpy as np
import av
from av.video.frame import VideoFrame

from caliscope.packets import FramePacket, PixelFormat

logger = logging.getLogger(__name__)


class FrameSource:
    """Forward-only frame access for recorded video files.

    next_frame() advances the stream and returns a FramePacket or None at EOF.
    When wanted_indices is provided at construction, frames not
    in that set are decoded (unavoidable with video codecs) but skipped without
    the costly BGR conversion — next_frame silently advances to the next wanted
    frame.

    Thread-safe for concurrent next_frame calls (internal lock), but a single
    instance is meant for one owner thread.
    """

    def __init__(
        self,
        video_directory: Path,
        cam_id: int,
        decode_threads: int = 0,
        wanted_indices: Set[int] | None = None,
        pixel_format: PixelFormat = PixelFormat.BGR,
    ) -> None:
        """Open cam_<cam_id>.mp4 in video_directory for forward-only reading.

        decode_threads caps the per-stream decode thread count; pass a share of
        the core budget (cpu_count // n_cameras) when several sources decode
        concurrently so they don't oversubscribe the cores. 0 lets the decoder
        use all cores, which is right for a single stream.

        wanted_indices, when provided, limits which frames get BGR conversion.
        next_frame silently skips unwanted frames. When None, every frame is
        wanted.
        """
        video_path = video_directory / f"cam_{cam_id}.mp4"
        self._open(
            video_path=video_path,
            cam_id=cam_id,
            decode_threads=decode_threads,
            wanted_indices=wanted_indices,
            pixel_format=pixel_format,
        )

    @classmethod
    def from_path(
        cls,
        video_path: Path,
        cam_id: int,
        decode_threads: int = 0,
        wanted_indices: Set[int] | None = None,
        pixel_format: PixelFormat = PixelFormat.BGR,
    ) -> Self:
        """Construct from an explicit video file path instead of the cam_N.mp4 convention."""
        instance = cls.__new__(cls)
        instance._open(
            video_path=video_path,
            cam_id=cam_id,
            decode_threads=decode_threads,
            wanted_indices=wanted_indices,
            pixel_format=pixel_format,
        )
        return instance

    def _open(
        self,
        video_path: Path,
        cam_id: int,
        decode_threads: int,
        wanted_indices: Set[int] | None,
        pixel_format: PixelFormat = PixelFormat.BGR,
    ) -> None:
        """Shared initialization. Call exactly once."""
        self.cam_id = cam_id
        self.video_path = video_path
        self._wanted: Set[int] | None = wanted_indices
        self._last_wanted: int | None = max(wanted_indices) if wanted_indices else None
        self._pixel_format = pixel_format

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._container = av.open(str(self.video_path))
        self._video_stream = self._container.streams.video[0]
        self._lock = Lock()
        self._frame_iterator: Iterator[VideoFrame] | None = None
        self._frame_index = -1

        if self._video_stream.time_base is None:
            raise ValueError(f"Video stream has no time_base: {self.video_path}")
        if self._video_stream.average_rate is None:
            raise ValueError(f"Video stream has no average_rate: {self.video_path}")

        self._time_base = float(self._video_stream.time_base)
        self.fps = float(self._video_stream.average_rate)
        self.size = (self._video_stream.width, self._video_stream.height)

        self._video_stream.thread_type = "AUTO"
        if decode_threads > 0:
            self._video_stream.codec_context.thread_count = decode_threads

        frame_count = self._video_stream.frames
        if frame_count == 0 and self._container.duration is not None:
            frame_count = int(self._container.duration / 1_000_000 * self.fps)
        self.frame_count = frame_count

        self._closed = False

    @property
    def start_frame_index(self) -> int:
        return 0

    @property
    def last_frame_index(self) -> int:
        return self.frame_count - 1

    def next_frame(self) -> FramePacket | None:
        """Return the next (wanted) frame as a FramePacket, or None at EOF.

        When wanted_indices was set at construction, silently advances past
        unwanted frames (decoding them but skipping BGR conversion). When no
        wanted_indices was set, every frame is returned. Stops early once the
        last wanted index has been passed.
        """
        with self._lock:
            if self._container is None:
                return None

            if self._frame_iterator is None:
                self._frame_iterator = self._container.decode(self._video_stream)

            try:
                while True:
                    frame = next(self._frame_iterator)
                    self._frame_index += 1
                    i = self._frame_index

                    # Past the last wanted frame — done.
                    if self._last_wanted is not None and i > self._last_wanted:
                        return None

                    # Skip unwanted frames without BGR conversion.
                    if self._wanted is not None and i not in self._wanted:
                        continue

                    frame_time = frame.pts * self._time_base if frame.pts is not None else 0.0

                    if self._pixel_format == PixelFormat.GRAY:
                        assert frame.format.name in ("yuv420p", "yuvj420p"), (
                            f"Expected yuv420p/yuvj420p for Y-plane extraction, got {frame.format.name}"
                        )
                        y_plane = frame.planes[0]
                        h = frame.height
                        w = frame.width
                        frame_array = np.ascontiguousarray(
                            np.frombuffer(y_plane, dtype=np.uint8).reshape(h, y_plane.line_size)[:, :w]
                        )
                    else:
                        frame_array = frame.to_ndarray(format="bgr24")

                    return FramePacket(
                        cam_id=self.cam_id,
                        frame_index=i,
                        frame_time=frame_time,
                        frame=frame_array,
                        pixel_format=self._pixel_format,
                    )

            except StopIteration:
                return None

    def close(self) -> None:
        """Release video container resources."""
        with self._lock:
            self._closed = True
            if self._container is not None:
                self._container.close()
                self._container = None  # type: ignore[assignment]
            self._frame_iterator = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            logger.warning(
                f"FrameSource for {self.video_path} was not closed properly. "
                "Use context manager or call close() explicitly."
            )
            self.close()
