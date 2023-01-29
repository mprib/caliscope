# The objective of this class is to create a dictionary of video streams that can
# be handed off to a synchronizer, which will then interact with the streams
# as though they were live video
# this is useful for two purposes:
#   1: future testing (don't have to keep recording live video)
#   2: future off-line processing of pre-recorded video.

import logging

LOG_FILE = r"log\recorded_stream.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)
from pathlib import Path
from queue import Queue
from threading import Thread
import numpy as np
import cv2
import pandas as pd


class RecordedStream:
    """Analogous to the live stream, this will place frames on a queue ("reel", probably need to 
    change that cutesy little thing). These can then be harvested and synchronized by a Synchronizer"""

    def __init__(self, port, directory):
        self.port = port
        self.directory = directory

        video_path = str(Path(self.directory, f"port_{port}.mp4"))
        synched_frames_history_path = str(Path(self.directory, f"frame_time_history.csv"))
        self.reel = Queue(-1)
        self.capture = cv2.VideoCapture(video_path)

        synched_frames_history = pd.read_csv(synched_frames_history_path)

        self.port_history = synched_frames_history[synched_frames_history["port"] == port]
        self.start_frame_index = self.port_history["frame_index"].min()
        self.last_frame_index = self.port_history["frame_index"].max()
        # self.shutter_sync = Queue(-1)

    def set_fps(self, fps):
        logging.info("No frame rate for recorded playback, push to synchronizer as rapidly as possible")

    def play_video(self):

        self.thread = Thread(target=self.play_video_worker, args=[], daemon=True)
        self.thread.start()

    def play_video_worker(self):
        """Places list of [frame_time, frame] on the reel for reading by a synchronizer,
        mimicking the behaviour of the LiveStream. 
        """
        frame_index = self.start_frame_index

        while True:
            
            # _ = self.shutter_sync.get()

            frame_time = self.port_history[self.port_history["frame_index"] == frame_index][
                "frame_time"
            ]
            frame_time = float(frame_time)
            success, frame = self.capture.read()

            if not success:
                break

            logging.debug(f"Placing frame on reel {self.port} for frame time: {frame_time} and frame index: {frame_index}")
            self.reel.put([frame_time, frame])
            frame_index += 1

            if frame_index > self.last_frame_index:
                logging.info(f"Ending recorded playback at port {self.port}")
                self.reel.put([-1, np.array([], dtype="uint8")])
                break


class RecordedStreamPool:
    
    def __init__(self, ports, directory):
        self.streams = {} 
        self.ports = ports 
        
        for port in ports:
            self.streams[port] = RecordedStream(port, directory)

    def play_videos(self):
        for port in self.ports:
            self.streams[port].play_video()
        

if __name__ == "__main__":
    import sys
    import time
    from calicam.cameras.synchronizer import Synchronizer
    
    repo = Path(__file__).parent.parent.parent
    print(repo)

    session_directory = Path(repo, "sessions", "iterative_adjustment", "recording")

    ports = [0,1]
    recorded_stream_pool = RecordedStreamPool(ports, session_directory)
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
    recorded_stream_pool.play_videos() 
    # recorded_stream = RecordedStream(port=port, directory=video_directory)
    # recorded_stream.start_video_to_reel()
    
    notification_q = Queue()
    syncr.synch_notice_subscribers.append(notification_q)

    while True:
        synched_frames_notice = notification_q.get()
        for port, frame_data in syncr.current_synched_frames.items():
            if frame_data:
                cv2.imshow(f"Port {port}", frame_data["frame"])

        key = cv2.waitKey(1)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break
