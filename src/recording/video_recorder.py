
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.synchronizer import Synchronizer

class VideoRecorder:

    def __init__(self, synchronizer):
        self.syncronizer = synchronizer

        # connect video recorder to synchronizer via a "bundle in" queue
        self.bundle_in_q = Queue()
        self.syncronizer.set_record_q(self.bundle_in_q)
        self.recording = False


    def build_video_writers(self, destination_folder):
        
        # create a dictionary of videowriters
        self.video_writers = {}
        for port, stream in self.syncronizer.streams.items():
            # path = str(Path(str(self.destination_path), "recordings", f"video_{port}.mp4"))
            path = str(Path(destination_folder, f"port_{port}.mp4"))
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            fps = 10 # I believe that this will work....
            frame_size = stream.camera.resolution
            
            logging.info(f"Recording destination is {path}")
            writer = cv2.VideoWriter(path, fourcc,fps, frame_size )
            self.video_writers[port] = writer

    def save_frame_worker(self, destination_folder):
        self.build_video_writers(destination_folder)

        while self.recording:
            frame_bundle = self.bundle_in_q.get() 

            for port, bundle in frame_bundle.items():
                
                frame = bundle["frame"]
                frame_index = bundle["frame_index"]
                frame_time = bundle["frame_time"]

                self.video_writers[port].write(frame)
                cv2.imshow(f"port: {port}", frame)
                
            key = cv2.waitKey(1)
            # if key == ord('q'):
            #     cv2.destroyAllWindows()
            #     break
        for port, bundle in frame_bundle.items():
            self.video_writers[port].release()

    def start_recording(self, destination_folder):
        self.recording = True
        self.recording_thread = Thread(target=self.save_frame_worker, args=[destination_folder,], daemon=True)
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

    syncr = Synchronizer(streams, fps_target=25)
    notification_q = Queue()
    syncr.subscribers.append(notification_q)

    repo = Path(__file__).parent.parent.parent
    print(repo)
    # destination_path = Path(repo, "sessions", "default_session", "recordings")

    video_recorder = VideoRecorder(syncr)

    video_recorder.start_recording(Path(Path(__file__).parent, "sample"))

    time.sleep(4)
    video_recorder.stop_recording()

    # while True:
    #     key = cv2.waitKey(1)

    #     if key == ord("q"):
    #         video_recorder.stop_recording() 
    #         break
    