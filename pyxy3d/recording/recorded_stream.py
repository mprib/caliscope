# The objective of this class is to create a dictionary of video streams that can
# be handed off to a synchronizer, which will then interact with the streams
# as though they were live video
# this is useful for two purposes:
#   1: future testing (don't have to keep recording live video)
#   2: future off-line processing of pre-recorded video.

import pyxy3d.logger
import logging

logger = pyxy3d.logger.get(__name__)
logger.setLevel(logging.INFO)

from pathlib import Path
from queue import Queue
from threading import Thread, Event
import toml

import cv2
from time import perf_counter, sleep
import pandas as pd
import numpy as np

from pyxy3d.interface import FramePacket, Tracker, Stream
from pyxy3d.trackers.tracker_enum import TrackerEnum
from pyxy3d.cameras.live_stream import Stream
from pyxy3d.cameras.camera_array import CameraData
from pyxy3d.configurator import Configurator


class RecordedStream(Stream):
    """
    Analogous to the live stream, this will place frames on a queue
    These can then be harvested and synchronized by a Synchronizer
    Within the stream, point detection occurs.
    """

    def __init__(
        self, camera: CameraData, directory: Path, fps_target:int=6, tracker: Tracker = None
    ):
        # self.port = port
        self.directory = directory
        self.camera = camera
        self.port = camera.port

        if tracker is not None:
            self.tracker = tracker
            self.track_points = True
        else:
            self.track_points = False

        video_path = str(Path(self.directory, f"port_{self.port}.mp4"))
        self.capture = cv2.VideoCapture(video_path)

        self.stop_event = Event()

        self.subscribers = []

        synched_frames_history_path = str(
            Path(self.directory, f"frame_time_history.csv")
        )
        synched_frames_history = pd.read_csv(synched_frames_history_path)

        self.port_history = synched_frames_history[
            synched_frames_history["port"] == self.port
        ]
       
        self.port_history['frame_index'] = self.port_history['frame_time'].rank(method='min').astype(int) - 1
        self.start_frame_index = self.port_history["frame_index"].min()
        self.last_frame_index = self.port_history["frame_index"].max()

        # initializing to something to avoid errors elsewhere
        self.frame_index = 0
        self.frame_time = 0
        self.set_fps_target(fps_target)

    def set_tracking_on(self, track: bool):
        if track:
            logger.info(f"Turning tracking on for recorded stream {self.port}")
            self.track_points.set()
        else:
            logger.info(f"Turning tracking off for recorded stream {self.port}")
            self.track_points.clear()

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

    def play_video(self):
        self.thread = Thread(target=self.process_frames, args=[], daemon=True)
        self.thread.start()

    def process_frames(self):
        """
        Places FramePacket on the out_q, mimicking the behaviour of the LiveStream.
        """

        self.frame_index = self.start_frame_index
        logger.info(f"Beginning playback of video for port {self.port}")
        while not self.stop_event.is_set():
            current_frame = self.port_history["frame_index"] == self.frame_index
            self.frame_time = self.port_history[current_frame]["frame_time"]
            self.frame_time = float(self.frame_time)

            spinlock_looped = False
            while len(self.subscribers) == 0 and not self.stop_event.is_set():
                if not spinlock_looped:
                    logger.info(f"Spinlock initiated at port {self.port}")
                    spinlock_looped = True
                sleep(0.5)
            if spinlock_looped == True:
                logger.info(f"Spinlock released at port {self.port}")

            if self.milestones is not None:
                sleep(self.wait_to_next_frame())

            success, self.frame = self.capture.read()

            if not success:
                break

            if self.track_points:
                self.point_data = self.tracker.get_points(self.frame, self.port, self.camera.rotation_count)
                draw_instructions = self.tracker.draw_instructions
            else:
                self.point_data = None
                draw_instructions = None


            frame_packet = FramePacket(
                port=self.port,
                frame_time=self.frame_time,
                frame=self.frame,
                # frame_index=self.frame_index,
                points=self.point_data,
                draw_instructions=draw_instructions
            )
            

            logger.debug(
                f"Placing frame on reel {self.port} for frame time: {self.frame_time} and frame index: {self.frame_index}"
            )

            for q in self.subscribers:
                q.put(frame_packet)

            # self.out_q.put(frame_packet)
            self.frame_index += 1

            if self.frame_index >= self.last_frame_index:
                logger.info(f"Ending recorded playback at port {self.port}")
                # time of -1 indicates end of stream
                frame_packet = FramePacket(self.port, -1, None, None)

                for q in self.subscribers:
                    q.put(frame_packet)
                break


class RecordedStreamPool:
    def __init__(
        self,
        directory: Path,
        config: Configurator,
        fps_target=6,
        tracker: Tracker = None,
    ):
        self.streams = {}
        self.camera_array = config.get_camera_array()

        for port, camera in self.camera_array.cameras.items():
            # tracker: Tracker = tracker.value()
            self.streams[port] = RecordedStream(
                camera, directory, fps_target=fps_target, tracker=tracker
            )

    def play_videos(self):
        for port, stream in self.streams.items():
            stream.play_video()


def get_configured_camera_data(config_path, intrinsics_only=True):
    """
    return a list of CameraData objects that is built from the config
    file that is found in the directory. This will be the same
    file where the mp4 files are located.
    """

    with open(config_path, "r") as f:
        config = toml.load(config_path)

    all_camera_data = {}
    for key, params in config.items():
        if key.startswith("cam"):
            if not params["ignore"]:
                port = params["port"]

                if intrinsics_only:
                    logger.info(
                        f"Adding intrinisic camera {port} to calibrated camera array..."
                    )

                    cam_data = CameraData(
                        port=port,
                        size=params["size"],
                        rotation_count=params["rotation_count"],
                        error=params["error"],
                        matrix=np.array(params["matrix"]),
                        distortions=np.array(params["distortions"]),
                        exposure=params["exposure"],
                        grid_count=params["grid_count"],
                        ignore=params["ignore"],
                        verified_resolutions=params["verified_resolutions"],
                    )
                else:
                    logger.info(
                        f"Adding intrinsic and extrinsic camera {port} to calibrated camera array..."
                    )
                    cam_data = CameraData(
                        port=port,
                        size=params["size"],
                        rotation_count=params["rotation_count"],
                        error=params["error"],
                        matrix=np.array(params["matrix"]),
                        distortions=np.array(params["distortions"]),
                        exposure=params["exposure"],
                        grid_count=params["grid_count"],
                        ignore=params["ignore"],
                        verified_resolutions=params["verified_resolutions"],
                        translation=np.array(params["translation"]),
                        rotation=np.array(params["rotation"]),
                    )

                all_camera_data[port] = cam_data

    return all_camera_data


if __name__ == "__main__":
    from pyxy3d.trackers.charuco_tracker import CharucoTracker
    from pyxy3d.cameras.synchronizer import Synchronizer
    from pyxy3d.calibration.charuco import Charuco

    from pyxy3d import __root__

    recording_directory = Path(__root__, "tests", "sessions", "post_monocal")

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )

    tracker = CharucoTracker(charuco)

    cameras = get_configured_camera_data(recording_directory)

    recorded_stream_pool = RecordedStreamPool(
        recording_directory, tracker_factory=tracker
    )
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
    recorded_stream_pool.play_videos()

    syncr.subscribe_to_streams()

    in_q = Queue(-1)
    syncr.subscribe_to_sync_packets(in_q)

    while not syncr.frames_complete:
        sleep(0.03)
        sync_packet = in_q.get()
        for port, frame_packet in sync_packet.frame_packets.items():
            if frame_packet:
                if frame_packet.frame_time == -1:
                    break  # end of frames
                cv2.imshow(f"Port {port}", frame_packet.frame_with_points)

        key = cv2.waitKey(1)

        if key == ord("q"):
            # cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
