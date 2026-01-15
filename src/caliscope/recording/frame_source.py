"""Raw frame access for recorded video files.

FrameSource provides synchronous frame reading and seeking with no threading,
queues, or tracking. It wraps PyAV for efficient video decoding.

This is infrastructure (I/O), not domain logic - hence placement in recording/
rather than core/.

PyAV/FFmpeg Quirks (lessons learned):
-------------------------------------
1. **PTS-to-frame-index calculation**: Must use `round()` not `int()`.
   The formula `frame.pts * time_base * fps` has floating point precision
   issues that cause `int()` to truncate incorrectly near frame boundaries.
   Example: PTS 285696 * time_base * fps = 557.8125 → int() gives 557, but
   round() correctly gives 558.

2. **Metadata frame count vs actual frames**: The container's frame count
   (from duration or stream.frames) may not match actually accessible frames.
   We scan keyframes at init to find the true last accessible frame index.

3. **skip_frame="NONKEY" corrupts decoder state**: After using skip_frame
   to scan keyframes, non-keyframe decoding fails silently. The container
   must be closed and reopened to restore normal behavior.

4. **Seeking near EOF**: PyAV's seek can fail silently near end of video,
   especially with B-frame encoded content. The keyframe index helps us
   know which frames are actually reachable.
"""

import logging
from pathlib import Path
from threading import Lock
from typing import Iterator, Self

import av
import numpy as np
from av.video.frame import VideoFrame

logger = logging.getLogger(__name__)


class FrameSource:
    """Raw frame access for recorded video files.

    Provides synchronous frame reading and seeking. Thread-safe for concurrent
    method calls (internal state protected by lock), but NOT thread-safe for
    access patterns: if multiple threads share a FrameSource, they must
    coordinate externally to avoid interleaved seek/read sequences.

    Typical usage: one owner thread, or explicit external synchronization.

    Note: get_frame() and get_nearest_keyframe() invalidate the sequential read
    position. After calling either, read_frame() behavior is undefined until
    the internal iterator is naturally exhausted or a new FrameSource is created.
    """

    def __init__(self, video_directory: Path, port: int) -> None:
        """Open a video file for frame access.

        Args:
            video_directory: Directory containing port_N.mp4 video files.
            port: Camera port number (used to construct filename).

        Raises:
            ValueError: If the video stream lacks required metadata.
            FileNotFoundError: If the video file doesn't exist.
        """
        self.port = port
        self.video_path = video_directory / f"port_{port}.mp4"

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._container = av.open(str(self.video_path))
        self._video_stream = self._container.streams.video[0]
        self._lock = Lock()
        self._frame_iterator: Iterator[VideoFrame] | None = None

        # Validate stream metadata
        if self._video_stream.time_base is None:
            raise ValueError(f"Video stream has no time_base: {self.video_path}")
        if self._video_stream.average_rate is None:
            raise ValueError(f"Video stream has no average_rate: {self.video_path}")

        self._time_base = float(self._video_stream.time_base)
        self.fps = float(self._video_stream.average_rate)
        self.size = (self._video_stream.width, self._video_stream.height)

        # Build keyframe index and find actual last frame
        # This solves PyAV/FFmpeg issues where metadata frame count doesn't match
        # accessible frames via seeking (especially near EOF with B-frames)
        self._keyframe_pts: list[int] = []
        self._keyframe_indices: list[int] = []
        self._actual_last_frame_index = self._build_frame_index()

        # frame_count from metadata (may differ from actual accessible frames)
        frame_count = self._video_stream.frames
        if frame_count == 0 and self._container.duration is not None:
            duration_seconds = self._container.duration / 1_000_000
            frame_count = int(duration_seconds * self.fps)
        self.frame_count = frame_count

        logger.debug(
            f"FrameSource for port {port}: metadata says {frame_count} frames, "
            f"actual last accessible frame is {self._actual_last_frame_index}, "
            f"found {len(self._keyframe_pts)} keyframes"
        )

        # Mark as successfully initialized - must be last line of __init__
        # If init fails, _closed won't exist and __del__ won't warn
        self._closed = False

    @property
    def start_frame_index(self) -> int:
        """First valid frame index (always 0 for raw video)."""
        return 0

    @property
    def last_frame_index(self) -> int:
        """Last valid frame index (actual accessible frame, not metadata estimate)."""
        return self._actual_last_frame_index

    def _build_frame_index(self) -> int:
        """Scan video to build keyframe index and find actual last frame.

        Uses skip_frame="NONKEY" to quickly iterate keyframes only.
        Records keyframe positions for potential future use in smarter seeking.

        Returns:
            Actual last accessible frame index.

        Note:
            Called during __init__ before _container could be set to None.
            Reopens container after scan because skip_frame corrupts decoder state.
        """
        # Container is guaranteed valid here - called from __init__ after open
        assert self._container is not None

        # Set decoder to skip non-keyframes for fast scanning
        self._video_stream.codec_context.skip_frame = "NONKEY"

        max_frame_idx = 0
        try:
            for frame in self._container.decode(self._video_stream):
                if frame.pts is None:
                    continue
                frame_idx = round(frame.pts * self._time_base * self.fps)
                self._keyframe_pts.append(frame.pts)
                self._keyframe_indices.append(frame_idx)
                max_frame_idx = max(max_frame_idx, frame_idx)
        except Exception as e:
            logger.warning(f"Error during keyframe scan: {e}")

        # Close and reopen container - skip_frame corrupts decoder state
        # such that non-keyframes become inaccessible after scanning
        self._container.close()
        self._container = av.open(str(self.video_path))
        self._video_stream = self._container.streams.video[0]
        self._frame_iterator = None

        # Find actual last accessible frame by seeking from last keyframe
        # The last keyframe may not be the last accessible frame
        if self._keyframe_indices:
            last_keyframe = self._keyframe_indices[-1]
            target_pts = int(last_keyframe / self.fps / self._time_base)
            self._container.seek(target_pts, stream=self._video_stream)

            # Decode forward from last keyframe to find true last frame
            for frame in self._container.decode(self._video_stream):
                if frame.pts is None:
                    continue
                frame_idx = round(frame.pts * self._time_base * self.fps)
                max_frame_idx = max(max_frame_idx, frame_idx)

            # Reset to beginning for normal use
            self._container.seek(0, stream=self._video_stream)

        return max_frame_idx

    def get_frame(self, frame_index: int) -> np.ndarray | None:
        """Seek to exact frame and return it as BGR numpy array.

        Uses PyAV's seek to nearest keyframe, then decodes forward to exact frame.
        O(keyframe_distance) complexity for compressed video.

        Args:
            frame_index: Target frame index to retrieve.

        Returns:
            Frame as BGR numpy array, or None if frame_index is out of bounds,
            seek fails, or frame not found.

        Note:
            Invalidates sequential read position. After calling this method,
            read_frame() behavior is undefined.
        """
        with self._lock:
            if self._container is None:
                return None

            # Bounds check - prevent PyAV's wrap-around behavior
            if frame_index < 0 or frame_index > self.last_frame_index:
                return None

            # Convert frame index to PTS (presentation timestamp)
            target_pts = int(frame_index / self.fps / self._time_base)
            self._container.seek(target_pts, stream=self._video_stream)

            # Decode frames until we reach or pass the target
            for frame in self._container.decode(self._video_stream):
                if frame.pts is None:
                    continue  # Skip frames without PTS
                frame_idx = round(frame.pts * self._time_base * self.fps)
                if frame_idx >= frame_index:
                    return frame.to_ndarray(format="bgr24")

            return None

    def get_nearest_keyframe(self, frame_index: int) -> tuple[np.ndarray | None, int]:
        """Seek to nearest keyframe at or before target.

        Fast seeking for scrubbing - returns the keyframe at or before target,
        without decoding forward to the exact frame. O(1) complexity for
        compressed video.

        Args:
            frame_index: Target frame index (will return keyframe at or before).

        Returns:
            Tuple of (frame_data, actual_frame_index).
            - frame_data: BGR numpy array, or None if out of bounds or seek fails.
            - actual_frame_index: Index of returned frame, or -1 if out of bounds
              or seek fails.

        Note:
            Invalidates sequential read position. After calling this method,
            read_frame() behavior is undefined.
        """
        with self._lock:
            if self._container is None:
                return None, -1

            # Bounds check - prevent PyAV's wrap-around behavior
            if frame_index < 0 or frame_index > self.last_frame_index:
                return None, -1

            # Convert frame index to PTS (presentation timestamp)
            target_pts = int(frame_index / self.fps / self._time_base)
            self._container.seek(target_pts, stream=self._video_stream)

            # Return the first frame after seek (the keyframe)
            for frame in self._container.decode(self._video_stream):
                if frame.pts is None:
                    continue
                actual_idx = round(frame.pts * self._time_base * self.fps)
                return frame.to_ndarray(format="bgr24"), actual_idx

            return None, -1

    def read_frame(self) -> np.ndarray | None:
        """Read next frame sequentially, returning BGR numpy array or None at EOF.

        Creates a new iterator on first call. Subsequent calls return frames
        in sequence until EOF.

        Returns:
            Frame as BGR numpy array, or None at end of file.

        Note:
            Position is undefined after get_frame() or get_nearest_keyframe() calls.
            For predictable sequential reading, use a fresh FrameSource or
            read all frames without seeking.
        """
        with self._lock:
            if self._container is None:
                return None

            if self._frame_iterator is None:
                self._frame_iterator = self._container.decode(self._video_stream)

            try:
                frame = next(self._frame_iterator)
                return frame.to_ndarray(format="bgr24")
            except StopIteration:
                return None

    def close(self) -> None:
        """Release video container resources."""
        with self._lock:
            self._closed = True
            if self._container is not None:
                self._container.close()
                self._container = None  # type: ignore[assignment]

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit - releases resources."""
        self.close()

    def __del__(self) -> None:
        """Destructor - warns if resources were not properly released.

        If _closed doesn't exist, __init__ failed and there's nothing to warn about.
        """
        if not getattr(self, "_closed", True):
            logger.warning(
                f"FrameSource for {self.video_path} was not closed properly. "
                "Use context manager or call close() explicitly."
            )
            self.close()
