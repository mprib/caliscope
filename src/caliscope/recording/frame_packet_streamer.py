"""Streamer for TrackedFrames from recorded video.

FramePacketStreamer wraps a FrameSource and FrameTimestamps, adding threading,
pub/sub broadcasting, and optional tracking. Forward-only: no seeking.

Notes:
------
1. **last_frame_index uses minimum of timestamps and source**: FrameTimestamps
   may have entries for frames that aren't actually accessible in the video
   file. We use min(timestamps, source) to ensure we don't try to seek beyond
   what's actually readable.
"""

import logging
from pathlib import Path
from queue import Queue
from threading import Condition, Event, Lock, Thread
from time import perf_counter, sleep
from typing import Literal

import numpy as np

from caliscope.packets import PixelFormat, TrackedFrame
from caliscope.recording.frame_source import FrameSource
from caliscope.recording.frame_timestamps import FrameTimestamps
from caliscope.task_manager.cancellation import CancellationToken
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


class FramePacketStreamer:
    """Streams TrackedFrames from recorded video to subscriber queues.

    Wraps FrameSource (video I/O) and FrameTimestamps (timing), adding:
    - Pub/sub broadcasting to multiple queues
    - Thread-safe subscriber management
    - Playback control (pause/unpause)
    - Optional tracker integration

    Thread Safety:
        Subscriber list is protected by a lock. The lock is released before
        blocking put() calls to avoid deadlock when queues are bounded.

    Duck-typed Interface:
        Exposes cam_id, subscribe(), unsubscribe() for Synchronizer compatibility.
    """

    def __init__(
        self,
        frame_source: FrameSource,
        frame_timestamps: FrameTimestamps,
        rotation_count: int = 0,
        tracker: Tracker | None = None,
        fps_target: float | None = None,
        end_behavior: Literal["stop", "pause"] = "stop",
    ) -> None:
        """Initialize the streamer.

        Args:
            frame_source: Video I/O wrapper (provides frames).
            frame_timestamps: Timing mapping (frame_index -> timestamp).
            rotation_count: Camera rotation (0, 1, 2, 3 for 0/90/180/270 degrees).
            tracker: Optional tracker for landmark detection.
            fps_target: Target playback FPS. None = unlimited (as fast as possible).
            end_behavior: What to do at last frame. "stop" broadcasts EOF and exits,
                "pause" auto-pauses for interactive scrubbing.
        """
        self._frame_source = frame_source
        self._frame_timestamps = frame_timestamps
        self._rotation_count = rotation_count
        self._tracker = tracker
        self._tracker_lock = Lock()  # Protect tracker swaps
        self._end_behavior = end_behavior

        # FPS targeting
        self._fps_target = fps_target
        self._milestones: np.ndarray | None = None
        if fps_target is not None:
            self._milestones = np.array([i / fps_target for i in range(int(fps_target))])

        # Thread-safe subscriber management
        self._subscribers: list[Queue] = []
        self._subscriber_lock = Lock()
        self._subscriber_condition = Condition(self._subscriber_lock)

        # Playback state
        self._pause_event = Event()
        self._pause_event.clear()

        # Current position
        self._frame_index = frame_timestamps.start_frame_index
        self._frame_time = 0.0

        # Thread management for play_video() convenience wrapper
        self._internal_token: CancellationToken | None = None
        self._thread: Thread | None = None

    # -------------------------------------------------------------------------
    # Properties (duck-typed interface for Synchronizer)
    # -------------------------------------------------------------------------

    @property
    def cam_id(self) -> int:
        """Camera identifier from underlying FrameSource."""
        return self._frame_source.cam_id

    @property
    def size(self) -> tuple[int, int]:
        """Frame dimensions (width, height)."""
        return self._frame_source.size

    @property
    def original_fps(self) -> float:
        """Original recording FPS from underlying video file."""
        return self._frame_source.fps

    @property
    def start_frame_index(self) -> int:
        """First valid frame index."""
        return self._frame_timestamps.start_frame_index

    @property
    def last_frame_index(self) -> int:
        """Last valid frame index (minimum of timestamps and source).

        FrameTimestamps may have entries for frames that aren't actually
        accessible in the video file. We use min() to ensure we don't try
        to seek beyond what's actually readable.
        """
        return min(self._frame_timestamps.last_frame_index, self._frame_source.last_frame_index)

    @property
    def frame_index(self) -> int:
        """Current frame index."""
        return self._frame_index

    @property
    def frame_time(self) -> float:
        """Current frame timestamp."""
        return self._frame_time

    def update_tracker(self, tracker: Tracker | None) -> None:
        """Swap the tracker reference. Thread-safe.

        Does NOT clean up the old tracker - caller is responsible for
        tracker lifecycle (construction, cleanup).

        Args:
            tracker: New tracker to use, or None to disable tracking.
        """
        with self._tracker_lock:
            self._tracker = tracker
        logger.info(f"Tracker updated for streamer at cam_id {self.cam_id}")

    # -------------------------------------------------------------------------
    # Pub/Sub (thread-safe)
    # -------------------------------------------------------------------------

    def subscribe(self, queue: Queue) -> None:
        """Add a queue to receive TrackedFrames.

        Thread-safe. Notifies waiting streamer if this is the first subscriber.
        """
        with self._subscriber_condition:
            if queue not in self._subscribers:
                logger.info(f"Adding subscriber to streamer at cam_id {self.cam_id}")
                self._subscribers.append(queue)
                self._subscriber_condition.notify()
            else:
                logger.warning(f"Attempted duplicate subscription to streamer at cam_id {self.cam_id}")

    def unsubscribe(self, queue: Queue) -> None:
        """Remove a queue from receiving TrackedFrames.

        Thread-safe.
        """
        with self._subscriber_lock:
            if queue in self._subscribers:
                logger.info(f"Removing subscriber from streamer at cam_id {self.cam_id}")
                self._subscribers.remove(queue)
            else:
                logger.warning(f"Attempted to unsubscribe non-existent queue at cam_id {self.cam_id}")

    def _broadcast(self, packet: TrackedFrame) -> None:
        """Send packet to all subscribers.

        Copies subscriber list while holding lock, then releases lock before
        calling put(). This prevents deadlock when queues are bounded and full.
        """
        with self._subscriber_lock:
            subscribers = self._subscribers.copy()
        for q in subscribers:
            q.put(packet)

    def _wait_for_subscribers(self, token: CancellationToken) -> bool:
        """Block until at least one subscriber exists or cancellation.

        Returns True if should continue, False if cancelled.
        """
        with self._subscriber_condition:
            logged = False
            while len(self._subscribers) == 0:
                if token.is_cancelled:
                    return False
                if not logged:
                    logger.info(f"Waiting for subscribers at cam_id {self.cam_id}")
                    logged = True
                self._subscriber_condition.wait(timeout=0.5)
            if logged:
                logger.info(f"Subscriber arrived at cam_id {self.cam_id}")
        return True

    # -------------------------------------------------------------------------
    # Playback Control
    # -------------------------------------------------------------------------

    def pause(self) -> None:
        """Pause playback."""
        logger.info(f"Pausing streamer at cam_id {self.cam_id}")
        self._pause_event.set()

    def unpause(self) -> None:
        """Resume playback."""
        logger.info(f"Unpausing streamer at cam_id {self.cam_id}")
        self._pause_event.clear()

    # -------------------------------------------------------------------------
    # FPS Timing
    # -------------------------------------------------------------------------

    def _wait_to_next_frame(self) -> float:
        """Calculate sleep duration to hit next frame milestone."""
        assert self._milestones is not None

        time = perf_counter()
        fractional_time = time % 1
        all_wait_times = self._milestones - fractional_time
        future_wait_times = all_wait_times[all_wait_times > 0]

        if len(future_wait_times) == 0:
            return 1 - fractional_time
        else:
            return future_wait_times[0]

    # -------------------------------------------------------------------------
    # Thread Lifecycle
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start playback in a new thread. Call stop() to terminate."""
        logger.info(f"Starting streamer for cam_id {self.cam_id}")
        self._internal_token = CancellationToken()
        self._thread = Thread(
            target=self.play_worker,
            args=[self._internal_token, None],
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop playback started via start()."""
        if self._internal_token is not None:
            self._internal_token.cancel()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info(f"Stopped streamer for cam_id {self.cam_id}")

    def close(self) -> None:
        """Release all resources."""
        self.stop()
        # Clean up tracker resources (threads, GPU memory) before closing frame source
        if self._tracker is not None:
            self._tracker.cleanup()
        self._frame_source.close()

    # -------------------------------------------------------------------------
    # Worker Loop
    # -------------------------------------------------------------------------

    def play_worker(self, token: CancellationToken, handle: TaskHandle | None = None) -> None:
        """Main playback loop. Places TrackedFrames on subscriber queues.

        Args:
            token: Cancellation token for cooperative shutdown.
            handle: Optional TaskHandle for progress reporting.
        """
        try:
            self._frame_index = self.start_frame_index

            logger.info(f"Beginning playback for cam_id {self.cam_id}")

            current_frame: np.ndarray | None = None

            while not token.is_cancelled:
                self._frame_time = self._frame_timestamps.get_time(self._frame_index)

                if not self._wait_for_subscribers(token):
                    break

                # FPS throttling
                if self._milestones is not None:
                    sleep(self._wait_to_next_frame())

                raw = self._frame_source.next_frame()
                current_pixel_format = PixelFormat.BGR
                if raw is not None:
                    self._frame_index = raw.frame_index
                    self._frame_time = raw.frame_time
                    current_frame = raw.frame
                    current_pixel_format = raw.pixel_format
                else:
                    current_frame = None

                # Handle EOF
                if current_frame is None:
                    logger.info(f"EOF reached at cam_id {self.cam_id}")
                    self._broadcast(
                        TrackedFrame(
                            cam_id=self.cam_id,
                            frame_index=-1,
                            frame_time=-1,
                            frame=None,
                            points=None,
                        )
                    )
                    break

                with self._tracker_lock:
                    tracker = self._tracker

                if tracker is not None:
                    point_data = tracker.get_points(current_frame, self.cam_id, self._rotation_count)
                    draw_instructions = tracker.scatter_draw_instructions
                else:
                    point_data = None
                    draw_instructions = None

                tracked = TrackedFrame(
                    cam_id=self.cam_id,
                    frame_index=self._frame_index,
                    frame_time=self._frame_time,
                    frame=current_frame,
                    points=point_data,
                    draw_instructions=draw_instructions,
                    pixel_format=current_pixel_format,
                )

                logger.debug(f"Broadcasting frame {self._frame_index} at cam_id {self.cam_id}")
                self._broadcast(tracked)

                if self._frame_index == self.last_frame_index:
                    if self._end_behavior == "stop":
                        logger.info(f"Reached last frame at cam_id {self.cam_id}")
                        self._broadcast(
                            TrackedFrame(
                                cam_id=self.cam_id,
                                frame_index=-1,
                                frame_time=-1,
                                frame=None,
                                points=None,
                            )
                        )
                        break
                    else:
                        self._pause_event.set()

                while self._pause_event.is_set() and not token.is_cancelled:
                    sleep(0.1)

        finally:
            logger.info(f"Streamer worker exiting for cam_id {self.cam_id}")


def create_streamer(
    video_directory: Path,
    cam_id: int,
    rotation_count: int = 0,
    tracker: Tracker | None = None,
    fps_target: float | None = None,
    end_behavior: Literal["stop", "pause"] = "stop",
    pixel_format: PixelFormat = PixelFormat.BGR,
) -> FramePacketStreamer:
    """Factory function to create a FramePacketStreamer.

    Convenience function that handles FrameSource and FrameTimestamps creation.

    Args:
        video_directory: Directory containing cam_N.mp4 and optionally timestamps.csv.
        cam_id: Camera identifier.
        rotation_count: Camera rotation (0, 1, 2, 3).
        tracker: Optional tracker for landmark detection.
        fps_target: Target FPS. None = unlimited.
        end_behavior: What to do at last frame. "stop" or "pause".
        pixel_format: Pixel format for frame decoding. Explicit opt-in; default is BGR.

    Returns:
        Configured FramePacketStreamer ready to start().
    """
    frame_source = FrameSource(video_directory, cam_id, pixel_format=pixel_format)

    timing_csv = video_directory / "timestamps.csv"
    if timing_csv.exists():
        frame_timestamps = FrameTimestamps.from_csv(timing_csv, cam_id)
    else:
        frame_timestamps = FrameTimestamps.inferred(frame_source.fps, frame_source.frame_count)

    return FramePacketStreamer(
        frame_source=frame_source,
        frame_timestamps=frame_timestamps,
        rotation_count=rotation_count,
        tracker=tracker,
        fps_target=fps_target,
        end_behavior=end_behavior,
    )
