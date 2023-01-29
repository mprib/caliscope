
import logging

LOG_FILE = r"log\video_recorder.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from pathlib import Path
from queue import Queue
from threading import Thread, Event
import cv2
import sys
import pandas as pd

from calicam.cameras.synchronizer import Synchronizer

class VideoRecorder:

    def __init__(self, synchronizer):
        self.syncronizer = synchronizer

        # build dict that will be stored to csv
        self.recording = False
        self.trigger_stop = Event()

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
        # connect video recorder to synchronizer via an "in" queue
        self.build_video_writers()

        self.frame_history = {"sync_index": [],
                               "port":[],
                               "frame_index":[],
                               "frame_time":[]}
        sync_index = 0

        self.synched_frames_in_q = Queue(-1)
        self.syncronizer.subscribe_to_synched_frames(self.synched_frames_in_q)       

        while not self.trigger_stop.is_set():
            synched_frames = self.synched_frames_in_q.get() 
            logging.debug("Pulling synched frames from record queue")

            for port, synched_frame_data in synched_frames.items():
                if synched_frame_data is not None:
                    # read in the data for this frame for this port
                    frame = synched_frame_data["frame"]
                    frame_index = synched_frame_data["frame_index"]
                    frame_time = synched_frame_data["frame_time"]

                    # store the frame
                    self.video_writers[port].write(frame)

                    # store to assocated data in the dictionary
                    self.frame_history["sync_index"].append(sync_index)
                    self.frame_history["port"].append(port)
                    self.frame_history["frame_index"].append(frame_index)
                    self.frame_history["frame_time"].append(frame_time)

                    # these two lines of code are just for ease of debugging 
                    cv2.imshow(f"port: {port}", frame)
                    key = cv2.waitKey(1)

            sync_index += 1
        self.trigger_stop.clear() # reset stop recording trigger
        self.syncronizer.release_synched_frames_q(self.synched_frames_in_q)

        # a proper release is strictly necessary to ensure file is readable
        for port, synched_frame_data in synched_frames.items():
            self.video_writers[port].release()

        self.store_frame_history()
    
    def store_frame_history(self):
        df = pd.DataFrame(self.frame_history)
        # TODO: #25 if file exists then change the name
        frame_hist_path = str(Path(self.destination_folder, "frame_time_history.csv"))
        logging.info(f"Storing frame history to {frame_hist_path}")
        df.to_csv(frame_hist_path, index = False, header = True)
        
         
    def start_recording(self, destination_folder):

        logging.info(f"All video data to be saved to {destination_folder}")

        self.destination_folder = destination_folder
        self.recording = True
        self.recording_thread = Thread(target=self.save_frame_worker, args=[], daemon=True)
        self.recording_thread.start() 


    def stop_recording(self):
        self.trigger_stop.set()


if __name__ == "__main__":

    import time

    from calicam.cameras.camera import Camera
    from calicam.cameras.live_stream import LiveStream
    from calicam.session import Session
    
    repo = str(Path(__file__)).split("src")[0]
    session_path = Path(repo, "sessions", "high_res_session")
    print(f"Config Path: {session_path}")
    session = Session(session_path)

    session.load_cameras()
    session.load_streams()
    session.adjust_resolutions()

    syncr = Synchronizer(session.streams, fps_target=50)
    notification_q = Queue()
    syncr.synch_notice_subscribers.append(notification_q)

    video_recorder = VideoRecorder(syncr)

    print(repo)
    video_path = Path(repo,"sessions", "high_res_session", "recording")
    video_recorder.start_recording(video_path)
    time.sleep(5)
    video_recorder.stop_recording()
