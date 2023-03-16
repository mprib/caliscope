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
import cv2
from time import perf_counter, sleep
import pandas as pd
import numpy as np

from pyxy3d.calibration.corner_tracker import CornerTracker
from pyxy3d.cameras.data_packets import FramePacket


class RecordedStream:
    """Analogous to the live stream, this will place frames on a queue ("reel", probably need to
    change that cutesy little thing). These can then be harvested and synchronized by a Synchronizer"""

    def __init__(self, port, directory, fps_target=6, charuco=None):
        self.port = port
        self.directory = directory

        if charuco is not None:
            self.tracker = CornerTracker(charuco)
            self.track_points = True
        else:
            self.track_points = False

        video_path = str(Path(self.directory, f"port_{port}.mp4"))
        self.capture = cv2.VideoCapture(video_path)

        self.push_to_out_q = Event()
        self.push_to_out_q.set()
        self.out_q = Queue(-1)
        self.stop_event = Event()

        synched_frames_history_path = str(
            Path(self.directory, f"frame_time_history.csv")
        )
        synched_frames_history = pd.read_csv(synched_frames_history_path)

        self.port_history = synched_frames_history[
            synched_frames_history["port"] == port
        ]
        self.start_frame_index = self.port_history["frame_index"].min()
        self.last_frame_index = self.port_history["frame_index"].max()

        # initializing to something to avoid errors elsewhere
        self.frame_index = 0
        self.frame_time = 0
        self.set_fps_target(fps_target)

    def set_fps_target(self, fps):
        self.fps = fps
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

        self.thread = Thread(target=self.worker, args=[], daemon=True)
        self.thread.start()

    def worker(self):
        """
        Places FramePacket on the out_q, mimicking the behaviour of the LiveStream.
        """

        self.frame_index = self.start_frame_index
        logger.info(f"Beginning playback of video for port {self.port}")
        while not self.stop_event.is_set():
            current_frame = self.port_history["frame_index"] == self.frame_index
            self.frame_time = self.port_history[current_frame]["frame_time"]
            self.frame_time = float(self.frame_time)
                
            sleep(self.wait_to_next_frame())
            

            success, self.frame = self.capture.read()

            if not success:
                break

            if self.track_points:
                self.point_data = self.tracker.get_points(self.frame)
            else:
                self.point_data = None

            frame_packet = FramePacket(
                port=self.port,
                frame_time=self.frame_time,
                frame=self.frame,
                frame_index=self.frame_index,
                points=self.point_data,
            )

            logger.debug(
                f"Placing frame on reel {self.port} for frame time: {self.frame_time} and frame index: {self.frame_index}"
            )
            self.out_q.put(frame_packet)
            self.frame_index += 1

            if self.frame_index >= self.last_frame_index:
                logger.info(f"Ending recorded playback at port {self.port}")
                # time of -1 indicates end of stream
                blank_packet = FramePacket(self.port, -1, None, None)
                self.out_q.put(blank_packet)
                break


class RecordedStreamPool:
    def __init__(self, ports, directory, fps_target=6, charuco=None):

        self.streams = {}
        self.ports = ports

        for port in ports:
            self.streams[port] = RecordedStream(port, directory, fps_target=fps_target, charuco=charuco)

    def play_videos(self):
        for port in self.ports:
            self.streams[port].play_video()


if __name__ == "__main__":
    from pyxy3d.cameras.synchronizer import Synchronizer
    from pyxy3d.calibration.charuco import Charuco
    
    
    from pyxy3d import __root__

    recording_directory = Path(__root__, "tests", "5_cameras", "recording")

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )

    # ports = [0, 1, 2, 3, 4]
    ports = [0,1,2, 3, 4]
    # ports = [0]
    recorded_stream_pool = RecordedStreamPool(ports, recording_directory, charuco=charuco)
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=6)
    recorded_stream_pool.play_videos()

    notification_q = Queue()
    syncr.sync_notice_subscribers.append(notification_q)

    while not syncr.frames_complete:
        synched_frames_notice = notification_q.get()
        for port, frame_packet in syncr.current_sync_packet.frame_packets.items():
            if frame_packet:
                cv2.imshow(f"Port {port}", frame_packet.frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
