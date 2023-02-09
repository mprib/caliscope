# The objective of this class is to create a dictionary of video streams that can
# be handed off to a synchronizer, which will then interact with the streams
# as though they were live video
# this is useful for two purposes:
#   1: future testing (don't have to keep recording live video)
#   2: future off-line processing of pre-recorded video.

import calicam.logger
import logging
logger = calicam.logger.get(__name__)
logger.setLevel(logging.INFO)

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
        self.out_q = Queue(-1)
        self.capture = cv2.VideoCapture(video_path)

        synched_frames_history = pd.read_csv(synched_frames_history_path)

        self.port_history = synched_frames_history[synched_frames_history["port"] == port]
        self.start_frame_index = self.port_history["frame_index"].min()
        self.last_frame_index = self.port_history["frame_index"].max()
        
        #initializing to something to avoid errors elsewhere
        self.frame_index = 0
        self.frame_time = 0 
        self.fps_target = None
        
    def set_fps_target(self, fps):
        self.fps_target = fps

    def play_video(self):

        self.thread = Thread(target=self.play_video_worker, args=[], daemon=True)
        self.thread.start()

    def play_video_worker(self):
        """Places list of [frame_time, frame] on the reel for reading by a synchronizer,
        mimicking the behaviour of the LiveStream. 
        """
        self.frame_index = self.start_frame_index
        logger.info(f"Beginning playback of video for port {self.port}")
        while True:

            self.frame_time = self.port_history[self.port_history["frame_index"] == self.frame_index][
                "frame_time"
            ]
            self.frame_time = float(self.frame_time)
            success, frame = self.capture.read()

            if not success:
                break

            logger.debug(f"Placing frame on reel {self.port} for frame time: {self.frame_time} and frame index: {self.frame_index}")
            self.out_q.put([self.frame_time, frame])
            self.frame_index += 1

            if self.frame_index >= self.last_frame_index:
                logger.info(f"Ending recorded playback at port {self.port}")
                self.out_q.put([-1, np.array([], dtype="uint8")])
                break

    def at_end_of_file(self):
        return self.frame_index == self.last_frame_index
class RecordedStreamPool:
    
    def __init__(self, ports, directory):
        self.streams = {} 
        self.ports = ports 
        
        for port in ports:
            self.streams[port] = RecordedStream(port, directory)

    def play_videos(self):
        self.thread = Thread(target=self.play_videos_worker, args=[],daemon=True)
        self.thread.start()
        
    def play_videos_worker(self):
        for port in self.ports:
            self.streams[port].play_video()
        
    def playback_complete(self):
        
        for port, stream in self.streams.items():
            if stream.at_end_of_file():
                
                return True
            else:
                return False
        
if __name__ == "__main__":
    from calicam.cameras.synchronizer import Synchronizer
    
    repo = Path(str(Path(__file__)).split("calicam")[0],"calicam")
    print(repo)

    # session_directory = Path(repo, "sessions", "iterative_adjustment", "recording")
    session_directory = Path(repo, "sessions", "5_cameras", "recording")

    ports = [0,1,2,3,4]
    recorded_stream_pool = RecordedStreamPool(ports, session_directory)
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
    recorded_stream_pool.play_videos()
     
    notification_q = Queue()
    syncr.synch_notice_subscribers.append(notification_q)

    while not recorded_stream_pool.playback_complete():
        synched_frames_notice = notification_q.get()
        for port, frame_data in syncr.current_synched_frames.items():
            if frame_data:
                cv2.imshow(f"Port {port}", frame_data["frame"])

        key = cv2.waitKey(1)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()