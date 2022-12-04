
import logging

LOG_FILE = r"log\video_recorder.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from pathlib import Path
from queue import Queue
from threading import Thread
import cv2
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.synchronizer import Synchronizer

class VideoRecorder:

    def __init__(self, synchronizer):
        self.syncronizer = synchronizer

        # connect video recorder to synchronizer via a "bundle in" queue
        self.recording = False

    def build_video_writers(self):
        
        # create a dictionary of videowriters
        self.video_writers = {}
        for port, stream in self.syncronizer.streams.items():

            path = str(Path(self.destination_folder, f"port_{port}.mp4"))
            logging.info(f"Building video writer for port {port}; recording to {path}")
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            fps = self.syncronizer.fps_target
            frame_size = stream.camera.resolution
            
            writer = cv2.VideoWriter(path, fourcc,fps, frame_size )
            self.video_writers[port] = writer


    def save_frame_worker(self):

        self.build_video_writers()
        # build dict that will be stored to csv
        self.bundle_history = {"bundle_index": [],
                               "port":[],
                               "frame_index":[],
                               "frame_time":[]}
        bundle_index = 0

        self.bundle_in_q = Queue()
        self.syncronizer.set_record_q(self.bundle_in_q)       

        while self.recording:
            frame_bundle = self.bundle_in_q.get() 
            logging.debug("Pulling bundle from record queue")

            for port, bundle in frame_bundle.items():
                if bundle is not None:
                    # read in the data for this frame for this port
                    frame = bundle["frame"]
                    frame_index = bundle["frame_index"]
                    frame_time = bundle["frame_time"]

                    # store the frame
                    self.video_writers[port].write(frame)

                    # store to assocated data in the dictionary
                    self.bundle_history["bundle_index"].append(bundle_index)
                    self.bundle_history["port"].append(port)
                    self.bundle_history["frame_index"].append(frame_index)
                    self.bundle_history["frame_time"].append(frame_time)

                    # these two lines of code are just for ease of debugging 
                    cv2.imshow(f"port: {port}", frame)
                    key = cv2.waitKey(1)

            bundle_index += 1

        self.syncronizer.release_record_q()

        # a proper release is strictly necessary to ensure file is readable
        for port, bundle in frame_bundle.items():
            self.video_writers[port].release()

        self.store_bundle_history()
    
    def store_bundle_history(self):
        df = pd.DataFrame(self.bundle_history)
        # TODO: #25 if file exists then change the name
        bundle_hist_path = str(Path(self.destination_folder, "bundle_history.csv"))
        logging.info(f"Storing bundle history to {bundle_hist_path}")
        df.to_csv(bundle_hist_path, index = False, header = True)
        
         
    def start_recording(self, destination_folder):

        logging.info(f"All video data to be saved to {destination_folder}")

        self.destination_folder = destination_folder
        self.recording = True
        self.recording_thread = Thread(target=self.save_frame_worker, args=[], daemon=True)
        self.recording_thread.start() 


    def stop_recording(self):
        self.recording = False



if __name__ == "__main__":

    import time

    from src.cameras.camera import Camera
    from src.cameras.video_stream import VideoStream

    cameras = []
    ports = [0, 1]
    for port in ports:
        cameras.append(Camera(port))

    streams = {}
    for cam in cameras:
        streams[cam.port] = VideoStream(cam)

    syncr = Synchronizer(streams, fps_target=30)
    notification_q = Queue()
    syncr.subscribers.append(notification_q)

    video_recorder = VideoRecorder(syncr)

    time.sleep(2)
    repo = Path(__file__).parent.parent.parent
    print(repo)
    video_path = Path(repo,"examples", "recordings", "sample1")
    video_recorder.start_recording(video_path)

    time.sleep(6)
    video_recorder.stop_recording()

    
    time.sleep(2)
    video_path = Path(repo, "examples", "recordings","sample2")
    video_recorder.start_recording(video_path)

    time.sleep(6)
    video_recorder.stop_recording()