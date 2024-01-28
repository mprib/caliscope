
import caliscope.logger
import logging

from pathlib import Path
from queue import Queue
from threading import Thread, Event
import rtoml

import cv2
from time import perf_counter, sleep
import pandas as pd
import numpy as np

from caliscope.packets import FramePacket, Tracker
from caliscope.cameras.camera_array import CameraData
from caliscope.configurator import Configurator

logger = caliscope.logger.get(__name__)
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
        port: int,
        # size: tuple = None,
        rotation_count: int = 0,
        fps_target: int = None,
        tracker: Tracker = None,
        break_on_last=True,
    ):
        # self.port = port
        self.directory = directory
        self.port = port
        self.rotation_count = rotation_count
        self.break_on_last = break_on_last  # stop while loop if end reached. Preferred behavior for automated file processing, not interactive frame selection

        self.tracker = tracker

        video_path = str(Path(self.directory, f"port_{self.port}.mp4"))
        self.capture = cv2.VideoCapture(video_path)

        # for playback, set the fps target to the actual
        self.original_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        if fps_target is None:
            fps_target = self.original_fps

        width =  int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (width,height)

        self.stop_event = Event()
        self._jump_q = Queue(maxsize=1)
        self._pause_event = Event()
        self._pause_event.clear()
        self.subscribers = []

        ############ PROCESS WITH TRUE TIME STAMPS IF AVAILABLE #########################
        synched_frames_history_path = Path(self.directory, "frame_time_history.csv")

        if synched_frames_history_path.exists():
            synched_frames_history = pd.read_csv(synched_frames_history_path)

            self.port_history = synched_frames_history[
                synched_frames_history["port"] == self.port
            ]

            self.port_history["frame_index"] = (
                self.port_history["frame_time"].rank(method="min").astype(int) - 1
            )

        ########### INFER TIME STAMP IF NOT AVAILABLE ####################################
        else:
            frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            mocked_port_history = {
                "frame_index": [i for i in range(0, frame_count)],
                "frame_time": [i / self.original_fps for i in range(0, frame_count)],
            }
            self.port_history = pd.DataFrame(mocked_port_history)

        # note that this is not simply 0 and frame count because the syncronized recording might start recording many frames into pulling from a camera
        # this is one of those unhappy artifacts that may be a good candidate for simplification in a future refactor
        self.start_frame_index = self.port_history["frame_index"].min()
        self.last_frame_index = self.port_history["frame_index"].max()

        # initialize properties
        self.frame_index = 0
        self.frame_time = 0
        self.set_fps_target(fps_target)

    # def set_tracking_on(self, track: bool):
    #     if track:
    #         logger.info(f"Turning tracking on for recorded stream {self.port}")
    #         self.track_points = 
    #     else:
    #         logger.info(f"Turning tracking off for recorded stream {self.port}")
    #         self.track_points.clear()

    def subscribe(self, queue: Queue):
        if queue not in self.subscribers:
            logger.info(f"Adding queue to subscribers at recorded stream {self.port}")
            self.subscribers.append(queue)
            logger.info(f"...now {len(self.subscribers)} subscriber(s) at {self.port}")
        else:
            logger.warn(
                f"Attempted to subscribe to recorded stream at port {self.port} twice"
            )

    def unsubscribe(self, queue: Queue):
        if queue in self.subscribers:
            logger.info(
                f"Removing subscriber from queue at recorded stream {self.port}"
            )
            self.subscribers.remove(queue)
            logger.info(
                f"{len(self.subscribers)} subscriber(s) remain at recorded stream {self.port}"
            )
        else:
            logger.warn(
                f"Attempted to unsubscribe to recorded stream that was not subscribed to\
                at port {self.port} twice"
            )

    def set_fps_target(self, fps):
        self.fps = fps
        if self.fps is None:
            self.milestones = None
        else:
            milestones = []
            for i in range(0, fps):
                milestones.append(i / fps)
            logger.info(f"Setting fps to {self.fps}")
            self.milestones = np.array(milestones)

    def wait_to_next_frame(self):
        """
        based on the next milestone time, return the time needed to sleep so that
        a frame read immediately after would occur when needed
        """

        time = perf_counter()
        fractional_time = time % 1
        all_wait_times = self.milestones - fractional_time
        future_wait_times = all_wait_times[all_wait_times > 0]

        if len(future_wait_times) == 0:
            return 1 - fractional_time
        else:
            return future_wait_times[0]

    def jump_to(self, frame_index: int):
        logger.info(f"Placing {frame_index} on jump q to reset capture position")
        self._jump_q.put(frame_index)

    def pause(self):
        logger.info(f"Pausing recorded stream at port {self.port}")
        self._pause_event.set()

    def unpause(self):
        logger.info(f"Unpausing recorded stream at port {self.port}")
        self._pause_event.clear()

    def play_video(self):
        logger.info(f"Initiating _play_worker for Camera {self.port}")
        self.thread = Thread(target=self._play_worker, args=[], daemon=False)
        self.thread.start()

    def _play_worker(self):
        """
        Places FramePacket on the out_q, mimicking the behaviour of the LiveStream.
        """

        self.frame_index = self.start_frame_index
        logger.info(f"Beginning playback of video for port {self.port}")

        while not self.stop_event.is_set():
            current_frame = self.port_history["frame_index"] == self.frame_index
            self.frame_time = self.port_history[current_frame]["frame_time"]
            self.frame_time = float(self.frame_time)

            ########## BEGIN NO SUBSCRIBERS SPINLOCK ##################
            spinlock_looped = False
            while len(self.subscribers) == 0 and not self.stop_event.is_set():
                if not spinlock_looped:
                    logger.info(f"Spinlock initiated at port {self.port}")
                    spinlock_looped = True
                sleep(0.5)
            if spinlock_looped:
                logger.info(f"Spinlock released at port {self.port}")
            ########## END NO SUBSCRIBERS SPINLOCK ##################

            if self.milestones is not None:
                sleep(self.wait_to_next_frame())
            logger.debug(
                f"about to read frame {self.frame_index} from capture at port {self.port}"
            )
            success, self.frame = self.capture.read()

            if not success:
                break


            if self.tracker is not None:
                self.point_data = self.tracker.get_points(
                    self.frame, self.port, self.rotation_count
                )
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
                f"Placing frame on q {self.port} for frame time: {self.frame_time} and frame index: {self.frame_index}"
            )

            for q in self.subscribers:
                q.put(frame_packet)

            # self.out_q.put(frame_packet)
            self.frame_index += 1

            if self.frame_index > self.last_frame_index and self.break_on_last:
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
            if not self.break_on_last and self.frame_index > self.last_frame_index:
                self.frame_index = self.last_frame_index
                self._pause_event.set()

            ############ SPIN LOCK FOR PAUSE ##################
            pause_logged = False
            while self._pause_event.is_set():
                # logger.info("I'm paused")
                if not pause_logged:
                    logger.info("Initiating Pause")
                    pause_logged = True

                if not self._jump_q.empty():
                    logger.info("New Value on jump queue, exiting pause spin lock")
                    break

                sleep(0.1)

            ############
            if not self._jump_q.empty():
                self.frame_index = self._jump_q.get()
                logger.info(
                    f"Setting port {self.port} capture object to frame index {self.frame_index}"
                )
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)

