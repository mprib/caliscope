# The objective of this class is to create a dictionary of video streams that can
# be handed off to a synchronizer, which will then interact with the streams
# as though they were live video
# this is useful for two purposes:
#   1: future testing (don't have to keep recording live video)
#   2: future off-line processing of pre-recorded video.

import logging

LOG_FILE = r"log\video_recorder.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import pandas as pd


class PlaybackStream:
    def __init__(self, port, directory):
        self.port = port
        self.directory = directory

        video_path = str(Path(self.directory, f"port_{port}.mp4"))
        bundle_history_path = str(Path(self.directory, f"bundle_history.csv"))
        self.reel = Queue(-1)
        self.capture = cv2.VideoCapture(video_path)

        self.bundle_history = pd.read_csv(bundle_history_path)

    def initiate_reel(self):

        self.thread = Thread(target=self.feed_reel, args=[], daemon=True)
        self.thread.start()

    def feed_reel(self):

        port_history = self.bundle_history[self.bundle_history["port"] == port]
        frame_index = port_history["frame_index"].min()

        while True:
            frame_time = port_history[port_history["frame_index"] == frame_index][
                "frame_time"
            ]
            frame_time = float(frame_time)
            success, frame = self.capture.read()
            
            if not success:
                break
            
            self.reel.put([frame_time, frame])
            
            frame_index+=1


if __name__ == "__main__":

    import time

    repo = Path(__file__).parent.parent.parent
    print(repo)
    video_directory = Path(repo, "examples", "recordings", "sample1")

    port = 1
    playback_stream = PlaybackStream(port=port, directory=video_directory)
    playback_stream.initiate_reel()
    
    while True:

        # time.sleep(.03)
        frame_time, reel_frame = playback_stream.reel.get()
        cv2.imshow(str(port), reel_frame)
        key = cv2.waitKey(1)
        print(frame_time)
