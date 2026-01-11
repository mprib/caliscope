import logging
from pathlib import Path
from queue import Queue
from threading import Condition, Event, Lock, Thread
from typing import Iterator

from caliscope.task_manager.cancellation import CancellationToken
from caliscope.task_manager.task_handle import TaskHandle
from time import perf_counter, sleep

import av
from av.video.frame import VideoFrame
import numpy as np
import pandas as pd
from caliscope.cameras.camera_array import CameraData
from caliscope.tracker import Tracker
from caliscope.packets import FramePacket

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RecordedStream:
    """
    Analogous to the live stream, this will place frames on a queue
    These can then be harvested and synchronized by a Synchronizer
    Within the stream, point detection occurs.
    """

    def __init__(
        self,
        directory: Path,
        camera: CameraData,
        fps_target: int | None = None,
        tracker: Tracker | None = None,
        break_on_last: bool = True,
    ):
        self.directory = directory
        self.camera = camera
        self.port = camera.port
        self.rotation_count = camera.rotation_count

        # stop while loop if end reached.
        # Preferred behavior for automated file processing, not interactive frame selection
        self.break_on_last = break_on_last

        self.tracker = tracker

        video_path = str(Path(self.directory, f"port_{self.port}.mp4"))

        # PyAV container and stream
        self._container = av.open(video_path)
        self._video_stream = self._container.streams.video[0]
        self._container_lock = Lock()
        self._frame_iterator: Iterator[VideoFrame] | None = None

        # These should always be set for a valid video stream
        if self._video_stream.time_base is None:
            raise ValueError(f"Video stream has no time_base: {video_path}")
        if self._video_stream.average_rate is None:
            raise ValueError(f"Video stream has no average_rate: {video_path}")

        self._time_base = float(self._video_stream.time_base)

        # for playback, set the fps target to the actual
        self.original_fps = int(float(self._video_stream.average_rate))
        if fps_target is None:
            fps_target = self.original_fps

        width = self._video_stream.width
        height = self._video_stream.height
        self.size = (width, height)

        # Jump request: (frame_index, exact) - latest value wins, overwrites pending
        self._pending_jump: tuple[int, bool] | None = None
        self._jump_lock = Lock()
        self._jump_condition = Condition(self._jump_lock)

        self._pause_event = Event()
        self._pause_event.clear()
        self.subscribers: list[Queue] = []

        # For play_video() convenience wrapper
        self._internal_token: CancellationToken | None = None
        self.thread: Thread | None = None

        ############ PROCESS WITH TRUE TIME STAMPS IF AVAILABLE #########################
        synched_frames_history_path = Path(self.directory, "frame_time_history.csv")

        if synched_frames_history_path.exists():
            synched_frames_history = pd.read_csv(synched_frames_history_path)

            # Explicitly create a copy to avoid SettingWithCopyWarning
            self.port_history = synched_frames_history[synched_frames_history["port"] == self.port].copy()
            self.port_history["frame_index"] = self.port_history["frame_time"].rank(method="min").astype(int) - 1

        ########### INFER TIME STAMP IF NOT AVAILABLE ####################################
        else:
            logger.info("Infering time stamps for frames based on capture data")
            # stream.frames may return 0 if unknown; fall back to duration-based estimate
            frame_count = self._video_stream.frames
            if frame_count == 0 and self._container.duration is not None:
                # duration is in microseconds
                duration_seconds = self._container.duration / 1_000_000
                frame_count = int(duration_seconds * self.original_fps)
            mocked_port_history = {
                "frame_index": [i for i in range(0, frame_count)],
                "frame_time": [i / self.original_fps for i in range(0, frame_count)],
            }
            self.port_history = pd.DataFrame(mocked_port_history)
            logger.info(f"frame_count: {frame_count}")
            logger.info(f"fps: {self.original_fps}")

        # note that this is not simply 0 and frame count because the syncronized recording might start recording many
        # frames into pulling from a camera
        # this is one of those unhappy artifacts that may be a good candidate for simplification in a future refactor
        self.start_frame_index = self.port_history["frame_index"].min()
        self.last_frame_index = self.port_history["frame_index"].max()

        # initialize properties
        self.frame_index = 0
        self.frame_time = 0.0
        self.set_fps_target(fps_target)

    def close(self) -> None:
        """Release video container resources."""
        with self._container_lock:
            if self._container is not None:
                self._container.close()
                self._container = None  # type: ignore[assignment]

    def subscribe(self, queue: Queue) -> None:
        if queue not in self.subscribers:
            logger.info(f"Adding queue to subscribers at recorded stream {self.port}")
            self.subscribers.append(queue)
            logger.info(f"...now {len(self.subscribers)} subscriber(s) at {self.port}")
        else:
            logger.warning(f"Attempted to subscribe to recorded stream at port {self.port} twice")

    def unsubscribe(self, queue: Queue) -> None:
        if queue in self.subscribers:
            logger.info(f"Removing subscriber from queue at recorded stream {self.port}")
            self.subscribers.remove(queue)
            logger.info(f"{len(self.subscribers)} subscriber(s) remain at recorded stream {self.port}")
        else:
            logger.warning(
                f"Attempted to unsubscribe to recorded stream that was not subscribed to\
                at port {self.port} twice"
            )

    def set_fps_target(self, fps: int | None) -> None:
        self.fps = fps
        if self.fps is None:
            self.milestones = None
        else:
            milestones = []
            for i in range(0, self.fps):
                milestones.append(i / self.fps)
            logger.info(f"Setting fps to {self.fps}")
            self.milestones = np.array(milestones)

    def wait_to_next_frame(self) -> float:
        """
        based on the next milestone time, return the time needed to sleep so that
        a frame read immediately after would occur when needed
        """
        assert self.milestones is not None  # Caller checks milestones before calling

        time = perf_counter()
        fractional_time = time % 1
        all_wait_times = self.milestones - fractional_time
        future_wait_times = all_wait_times[all_wait_times > 0]

        if len(future_wait_times) == 0:
            return 1 - fractional_time
        else:
            return future_wait_times[0]

    def jump_to(self, frame_index: int, exact: bool = True) -> None:
        """Request a seek to the specified frame.

        Args:
            frame_index: Target frame to seek to.
            exact: If True (default), decode to exact frame. If False, seek to
                   nearest keyframe only (faster, good for scrubbing during drag).

        Note: If multiple jump requests arrive before the worker processes them,
        only the latest request is honored (previous pending jumps are dropped).
        """
        with self._jump_condition:
            logger.info(f"Setting pending jump to frame {frame_index} (exact={exact})")
            self._pending_jump = (frame_index, exact)
            self._jump_condition.notify()

    def _has_pending_jump(self) -> bool:
        """Check if there's a pending jump request."""
        with self._jump_lock:
            return self._pending_jump is not None

    def _take_pending_jump(self) -> tuple[int, bool] | None:
        """Take and clear the pending jump request."""
        with self._jump_lock:
            jump = self._pending_jump
            self._pending_jump = None
            return jump

    def pause(self) -> None:
        logger.info(f"Pausing recorded stream at port {self.port}")
        self._pause_event.set()

    def unpause(self) -> None:
        logger.info(f"Unpausing recorded stream at port {self.port}")
        self._pause_event.clear()

    def play_video(self) -> None:
        """Start playback in a new thread. Call stop() to terminate."""
        logger.info(f"Initiating play_worker for Camera {self.port}")
        self._internal_token = CancellationToken()
        self.thread = Thread(target=self.play_worker, args=[self._internal_token, None], daemon=False)
        self.thread.start()

    def stop(self) -> None:
        """Stop playback started via play_video()."""
        if self._internal_token is not None:
            self._internal_token.cancel()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        logger.info(f"Stopped playback for port {self.port}")

    def _seek_to_frame(self, target_frame_index: int) -> np.ndarray | None:
        """Seek to exact frame and return it as BGR numpy array.

        Uses PyAV's seek to nearest keyframe, then decodes forward to exact frame.
        Returns None if seek fails or frame not found.
        """
        with self._container_lock:
            if self._container is None:
                return None

            # Convert frame index to PTS (presentation timestamp)
            target_pts = int(target_frame_index / self.original_fps / self._time_base)
            self._container.seek(target_pts, stream=self._video_stream)

            # Decode frames until we reach or pass the target
            for frame in self._container.decode(self._video_stream):
                if frame.pts is None:
                    continue  # Skip frames without PTS
                frame_idx = int(frame.pts * self._time_base * self.original_fps)
                if frame_idx >= target_frame_index:
                    return frame.to_ndarray(format="bgr24")

            return None

    def _seek_to_keyframe(self, target_frame_index: int) -> tuple[np.ndarray | None, int]:
        """Seek to nearest keyframe and return it with actual frame index.

        Fast seeking for scrubbing - returns the keyframe at or before target,
        without decoding forward to the exact frame.

        Returns:
            Tuple of (frame_data, actual_frame_index). Frame data is None if seek fails.
        """
        with self._container_lock:
            if self._container is None:
                return None, target_frame_index

            # Convert frame index to PTS (presentation timestamp)
            target_pts = int(target_frame_index / self.original_fps / self._time_base)
            self._container.seek(target_pts, stream=self._video_stream)

            # Return the first frame after seek (the keyframe)
            for frame in self._container.decode(self._video_stream):
                if frame.pts is None:
                    continue
                actual_idx = int(frame.pts * self._time_base * self.original_fps)
                return frame.to_ndarray(format="bgr24"), actual_idx

            return None, target_frame_index

    def _read_next_frame(self) -> np.ndarray | None:
        """Read the next frame from the iterator, returning BGR numpy array or None at EOF."""
        with self._container_lock:
            if self._container is None:
                return None

            if self._frame_iterator is None:
                self._frame_iterator = self._container.decode(self._video_stream)

            try:
                frame = next(self._frame_iterator)
                return frame.to_ndarray(format="bgr24")
            except StopIteration:
                return None

    def _reset_iterator(self) -> None:
        """Reset the frame iterator after a seek."""
        with self._container_lock:
            if self._container is not None:
                self._frame_iterator = self._container.decode(self._video_stream)

    def play_worker(self, token: CancellationToken, handle: TaskHandle | None = None) -> None:
        """
        Places FramePacket on the out_q, mimicking the behaviour of the LiveStream.
        """
        try:
            self.frame_index = self.start_frame_index

            # Seek to start frame if not starting at 0
            if self.start_frame_index > 0:
                self._seek_to_frame(self.start_frame_index)
            self._reset_iterator()

            logger.info(f"Beginning playback of video for port {self.port}")

            # Flag to skip read when we already have the frame from a seek
            skip_read = False

            while not token.is_cancelled:
                current_frame = self.port_history["frame_index"] == self.frame_index
                self.frame_time = self.port_history[current_frame]["frame_time"]
                self.frame_time = float(self.frame_time.iloc[0])

                ########## BEGIN NO SUBSCRIBERS SPINLOCK ##################
                spinlock_looped = False
                while len(self.subscribers) == 0 and not token.is_cancelled:
                    if not spinlock_looped:
                        logger.info(f"Spinlock initiated at port {self.port}")
                        spinlock_looped = True
                    token.sleep_unless_cancelled(0.5)
                if spinlock_looped:
                    logger.info(f"Spinlock released at port {self.port}")
                ########## END NO SUBSCRIBERS SPINLOCK ##################

                if self.milestones is not None:
                    sleep(self.wait_to_next_frame())

                # Read next frame unless we already have it from a seek
                if skip_read:
                    skip_read = False
                else:
                    self.frame = self._read_next_frame()

                if self.frame is None:
                    # Iterator exhausted - send EOF marker before exiting
                    logger.info(f"Iterator exhausted at port {self.port}")
                    eof_packet = FramePacket(
                        port=self.port,
                        frame_index=-1,
                        frame_time=-1,
                        frame=None,
                        points=None,
                    )
                    for q in self.subscribers:
                        q.put(eof_packet)
                    break

                if self.tracker is not None:
                    self.point_data = self.tracker.get_points(self.frame, self.port, self.rotation_count)
                    draw_instructions = self.tracker.scatter_draw_instructions
                else:
                    self.point_data = None
                    draw_instructions = None

                frame_packet = FramePacket(
                    port=self.port,
                    frame_index=self.frame_index,
                    frame_time=self.frame_time,
                    frame=self.frame,
                    points=self.point_data,
                    draw_instructions=draw_instructions,
                )

                logger.debug(
                    f"Placing frame on q {self.port} for frame time: {self.frame_time} "
                    f"and frame index: {self.frame_index}"
                )

                for q in self.subscribers:
                    q.put(frame_packet)

                if self.frame_index == self.last_frame_index and self.break_on_last:
                    logger.info(f"Ending recorded playback at port {self.port}")
                    # time of -1 indicates end of stream
                    frame_packet = FramePacket(
                        port=self.port,
                        frame_index=-1,
                        frame_time=-1,
                        frame=None,
                        points=None,
                    )

                    for q in self.subscribers:
                        q.put(frame_packet)
                    break

                ############ Autopause if last frame and in playback mode (i.e. break_on_last == False)
                if not self.break_on_last and self.frame_index == self.last_frame_index:
                    self._pause_event.set()

                ############ SPIN LOCK FOR PAUSE ##################
                pause_logged = False
                while self._pause_event.is_set() and not token.is_cancelled:
                    if not pause_logged:
                        logger.info("Initiating Pause")
                        pause_logged = True

                    if self._has_pending_jump():
                        logger.info("Pending jump detected, exiting pause spin lock")
                        break

                    token.sleep_unless_cancelled(0.1)
                #######################################################
                pending = self._take_pending_jump()
                if pending is not None:
                    target_index, exact = pending
                    logger.info(f"Processing jump to frame {target_index} (exact={exact}) at port {self.port}")

                    if exact:
                        # Exact seeking - decode forward to precise frame
                        frame = self._seek_to_frame(target_index)
                        if frame is not None:
                            self.frame = frame
                            self.frame_index = target_index
                            skip_read = True
                    else:
                        # Keyframe seeking - fast, returns nearest keyframe
                        frame, actual_index = self._seek_to_keyframe(target_index)
                        if frame is not None:
                            self.frame = frame
                            self.frame_index = actual_index
                            skip_read = True

                    # Reset iterator to continue from current position
                    self._reset_iterator()
                else:
                    # Increment for next iteration (only if not jumping)
                    self.frame_index += 1
        finally:
            self.close()
