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
    def __init__(self, port, directory):
        self.port = port
        self.directory = directory

        video_path = str(Path(self.directory, f"port_{port}.mp4"))
        bundle_history_path = str(Path(self.directory, f"bundle_history.csv"))
        self.reel = Queue(-1)
        self.capture = cv2.VideoCapture(video_path)

        self.bundle_history = pd.read_csv(bundle_history_path)

    def start_video_to_reel(self):

        self.thread = Thread(target=self.video_to_reel, args=[], daemon=True)
        self.thread.start()

    def video_to_reel(self):
        """Places list of [frame_time, frame] on the reel for reading by a synchronizer,
        mimicking the behaviour of the LiveStream. 
        """

        port_history = self.bundle_history[self.bundle_history["port"] == port]
        frame_index = port_history["frame_index"].min()
        last_frame = port_history["frame_index"].max()

        while True:
            frame_time = port_history[port_history["frame_index"] == frame_index][
                "frame_time"
            ]
            frame_time = float(frame_time)
            success, frame = self.capture.read()

            # print(frame_time)

            if not success:
                break

            self.reel.put([frame_time, frame])

            frame_index += 1

            if frame_index > last_frame:
                self.reel.put([-1, np.array([], dtype="uint8")])
                break


if __name__ == "__main__":

    import time

    repo = Path(__file__).parent.parent.parent
    print(repo)
    video_directory = Path(repo, "examples", "recordings", "sample2")

    port = 1
    recorded_stream = RecordedStream(port=port, directory=video_directory)
    recorded_stream.start_video_to_reel()

    while True:
        # time.sleep(0.03)
        frame_time, reel_frame = recorded_stream.reel.get()
        if frame_time == -1:
            cv2.destroyAllWindows()
            break
        
        cv2.imshow(str(port), reel_frame)
        key = cv2.waitKey(1)
        # print(frame_time)
