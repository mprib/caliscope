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
from queue import Queue
from pathlib import Path
import pandas as pd
import cv2


class PlaybackStream:
    def __init__(self, port, directory):
        self.port = port
        self.directory = directory
        pass
    
    

if __name__ == "__main__":

    import time

    repo = Path(__file__).parent.parent.parent
    print(repo)
    video_path = Path(repo, "examples", "recordings", "sample1")

    port = 1
    playback_stream = PlaybackStream(port=port, directory=video_path)

    reel = Queue(-1)

    mp4_file = str(Path(video_path, f"port_{port}.mp4"))
    bundle_history_path =str(Path(video_path, f"bundle_history.csv")) 
    
    bundle_history = pd.read_csv(bundle_history_path)

    port_history = bundle_history[bundle_history["port"]==port]


    frame_index = port_history["frame_index"].min()

    cap = cv2.VideoCapture(mp4_file)

    while True:

        success, frame = cap.read()
        key = cv2.waitKey(1)

        if not success:
            break

        if key == ord("q"):
            break


        frame_time = port_history[port_history["frame_index"]==frame_index]["frame_time"]
        frame_time = float(frame_time)


        reel.put([frame_time, frame])
        # time.sleep(.03)
        frame_time, reel_frame = reel.get()
        cv2.imshow(str(port), reel_frame)
        print(frame_time)
        frame_index +=1