"""Streamer for FramePackets from recorded video.

FramePacketStreamer wraps a FrameSource and FrameTimestamps, adding threading,
pub/sub broadcasting, and optional tracking. It's the streaming layer on top
of the raw video I/O.

Seeking and Pause Handling (lessons learned):
---------------------------------------------
1. **Seek failure must not kill the streamer**: When get_frame() returns None
   (e.g., PyAV can't decode near EOF), we must set skip_read=True to stay at
   the current position. Otherwise, the next read_frame() call has undefined
   behavior after a seek and will likely return None, causing premature exit.

2. **Condition variable for responsive pause loop**: The pause loop uses
   _seek_condition.wait() instead of plain sleep. This allows seek_to() calls
   to wake the loop immediately via notify(), making scrubbing responsive.
   Without this, there's up to 0.1s latency on each seek while paused.

3. **last_frame_index uses minimum of timestamps and source**: FrameTimestamps
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

from caliscope.packets import FramePacket
from caliscope.recording.frame_source import FrameSource
from caliscope.recording.frame_timestamps import FrameTimestamps
from caliscope.task_manager.cancellation import CancellationToken
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


class FramePacketStreamer:
    """Streams FramePackets from recorded video to subscriber queues.

    Wraps FrameSource (video I/O) and FrameTimestamps (timing), adding:
    - Pub/sub broadcasting to multiple queues
    - Thread-safe subscriber management
    - Playback control (pause/unpause, seeking)
    - Optional tracker integration

    Thread Safety:
        Subscriber list is protected by a lock. The lock is released before
        blocking put() calls to avoid deadlock when queues are bounded.

    Duck-typed Interface:
        Exposes port, subscribe(), unsubscribe() for Synchronizer compatibility.
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

        # Seek request: (frame_index, exact) - latest wins
        self._pending_seek: tuple[int, bool] | None = None
        self._seek_lock = Lock()
        self._seek_condition = Condition(self._seek_lock)

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
    def port(self) -> int:
        """Camera port from underlying FrameSource."""
        return self._frame_source.port

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

    # -------------------------------------------------------------------------
    # Pub/Sub (thread-safe)
    # -------------------------------------------------------------------------

    def subscribe(self, queue: Queue) -> None:
        """Add a queue to receive FramePackets.

        Thread-safe. Notifies waiting streamer if this is the first subscriber.
        """
        with self._subscriber_condition:
            if queue not in self._subscribers:
                logger.info(f"Adding subscriber to streamer at port {self.port}")
                self._subscribers.append(queue)
                self._subscriber_condition.notify()
            else:
                logger.warning(f"Attempted duplicate subscription to streamer at port {self.port}")

    def unsubscribe(self, queue: Queue) -> None:
        """Remove a queue from receiving FramePackets.

        Thread-safe.
        """
        with self._subscriber_lock:
            if queue in self._subscribers:
                logger.info(f"Removing subscriber from streamer at port {self.port}")
                self._subscribers.remove(queue)
            else:
                logger.warning(f"Attempted to unsubscribe non-existent queue at port {self.port}")

    def _broadcast(self, packet: FramePacket) -> None:
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
                    logger.info(f"Waiting for subscribers at port {self.port}")
                    logged = True
                self._subscriber_condition.wait(timeout=0.5)
            if logged:
                logger.info(f"Subscriber arrived at port {self.port}")
        return True

    # -------------------------------------------------------------------------
    # Playback Control
    # -------------------------------------------------------------------------

    def pause(self) -> None:
        """Pause playback."""
        logger.info(f"Pausing streamer at port {self.port}")
        self._pause_event.set()

    def unpause(self) -> None:
        """Resume playback."""
        logger.info(f"Unpausing streamer at port {self.port}")
        self._pause_event.clear()

    def seek_to(self, frame_index: int, precise: bool = True) -> None:
        """Request a seek to the specified frame.

        Args:
            frame_index: Target frame to seek to.
            precise: If True, decode to exact frame (slower). If False, seek to
                nearest keyframe only (faster for scrubbing).

        Note: If multiple seek requests arrive before processing, only the
        latest is honored (previous pending seeks are dropped).
        """
        with self._seek_condition:
            logger.info(f"Setting pending seek to frame {frame_index} (precise={precise})")
            self._pending_seek = (frame_index, precise)
            self._seek_condition.notify()

    def _has_pending_seek(self) -> bool:
        """Check for pending seek request."""
        with self._seek_lock:
            return self._pending_seek is not None

    def _take_pending_seek(self) -> tuple[int, bool] | None:
        """Take and clear the pending seek request."""
        with self._seek_lock:
            seek = self._pending_seek
            self._pending_seek = None
            return seek

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
        logger.info(f"Starting streamer for port {self.port}")
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
        logger.info(f"Stopped streamer for port {self.port}")

    def close(self) -> None:
        """Release all resources."""
        self.stop()
        self._frame_source.close()

    # -------------------------------------------------------------------------
    # Worker Loop
    # -------------------------------------------------------------------------

    def play_worker(self, token: CancellationToken, handle: TaskHandle | None = None) -> None:
        """Main playback loop. Places FramePackets on subscriber queues.

        Args:
            token: Cancellation token for cooperative shutdown.
            handle: Optional TaskHandle for progress reporting.
        """
        try:
            self._frame_index = self.start_frame_index

            # Seek to start frame if not starting at 0
            if self.start_frame_index > 0:
                self._frame_source.get_frame(self.start_frame_index)

            logger.info(f"Beginning playback for port {self.port}")

            # Track current frame data
            current_frame: np.ndarray | None = None
            skip_read = False

            while not token.is_cancelled:
                # Update frame time from timestamps
                self._frame_time = self._frame_timestamps.get_time(self._frame_index)

                # Wait for subscribers (Condition-based, not spinlock)
                if not self._wait_for_subscribers(token):
                    break

                # FPS throttling
                if self._milestones is not None:
                    sleep(self._wait_to_next_frame())

                # Read next frame unless we have it from a seek
                if skip_read:
                    skip_read = False
                else:
                    current_frame = self._frame_source.read_frame()

                # Handle EOF
                if current_frame is None:
                    logger.info(f"EOF reached at port {self.port}")
                    eof_packet = FramePacket(
                        port=self.port,
                        frame_index=-1,
                        frame_time=-1,
                        frame=None,
                        points=None,
                    )
                    self._broadcast(eof_packet)
                    break

                # Track points if tracker attached
                if self._tracker is not None:
                    point_data = self._tracker.get_points(current_frame, self.port, self._rotation_count)
                    draw_instructions = self._tracker.scatter_draw_instructions
                else:
                    point_data = None
                    draw_instructions = None

                # Create and broadcast packet
                frame_packet = FramePacket(
                    port=self.port,
                    frame_index=self._frame_index,
                    frame_time=self._frame_time,
                    frame=current_frame,
                    points=point_data,
                    draw_instructions=draw_instructions,
                )

                logger.debug(f"Broadcasting frame {self._frame_index} at port {self.port}")
                self._broadcast(frame_packet)

                # Handle last frame
                if self._frame_index == self.last_frame_index:
                    if self._end_behavior == "stop":
                        logger.info(f"Reached last frame at port {self.port}")
                        eof_packet = FramePacket(
                            port=self.port,
                            frame_index=-1,
                            frame_time=-1,
                            frame=None,
                            points=None,
                        )
                        self._broadcast(eof_packet)
                        break
                    else:
                        # Auto-pause at end for interactive mode
                        self._pause_event.set()

                # Handle pause (check for seeks while paused)
                while self._pause_event.is_set() and not token.is_cancelled:
                    if self._has_pending_seek():
                        break
                    # Wait on condition variable - wakes immediately when seek_to() calls notify()
                    with self._seek_condition:
                        self._seek_condition.wait(timeout=0.1)

                # Process pending seek
                pending = self._take_pending_seek()
                if pending is not None:
                    target_index, precise = pending
                    logger.info(f"Processing seek to frame {target_index} (precise={precise}) at port {self.port}")

                    if precise:
                        frame = self._frame_source.get_frame(target_index)
                        if frame is not None:
                            current_frame = frame
                            self._frame_index = target_index
                            skip_read = True
                        else:
                            # Seek failed - stay at current position, don't call read_frame()
                            logger.warning(f"Seek to frame {target_index} failed at port {self.port}")
                            skip_read = True
                    else:
                        frame, actual_index = self._frame_source.get_nearest_keyframe(target_index)
                        if frame is not None:
                            current_frame = frame
                            self._frame_index = actual_index
                            skip_read = True
                        else:
                            # Fast seek failed - stay at current position
                            logger.warning(f"Fast seek to frame {target_index} failed at port {self.port}")
                            skip_read = True
                else:
                    # Increment for next iteration (only if not seeking)
                    self._frame_index += 1

        finally:
            logger.info(f"Streamer worker exiting for port {self.port}")


def create_streamer(
    video_directory: Path,
    port: int,
    rotation_count: int = 0,
    tracker: Tracker | None = None,
    fps_target: float | None = None,
    end_behavior: Literal["stop", "pause"] = "stop",
) -> FramePacketStreamer:
    """Factory function to create a FramePacketStreamer.

    Convenience function that handles FrameSource and FrameTimestamps creation.

    Args:
        video_directory: Directory containing port_N.mp4 and optionally frame_time_history.csv.
        port: Camera port number.
        rotation_count: Camera rotation (0, 1, 2, 3).
        tracker: Optional tracker for landmark detection.
        fps_target: Target FPS. None = unlimited.
        end_behavior: What to do at last frame. "stop" or "pause".

    Returns:
        Configured FramePacketStreamer ready to start().
    """
    frame_source = FrameSource(video_directory, port)

    timing_csv = video_directory / "frame_time_history.csv"
    if timing_csv.exists():
        frame_timestamps = FrameTimestamps.from_csv(timing_csv, port)
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
