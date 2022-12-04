import cv2
from pathlib import Path

from queue import Queue
from threading import Thread
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.synchronizer import Synchronizer

from src.cameras.camera import Camera
from src.cameras.video_stream import VideoStream

ports = [0, 1]

cameras = []
for port in ports:
    cameras.append(Camera(port))

streams = {}
for cam in cameras:
    streams[cam.port] = VideoStream(cam)

syncr = Synchronizer(streams, fps_target=25)

bundle_in_q = Queue()
syncr.set_record_q(bundle_in_q)

repo = Path(__file__).parent.parent.parent
print(repo)
fourcc = cv2.VideoWriter_fourcc(*"MP4V")


def frame_writer():

    video_writers = {}
    for port, stream in syncr.streams.items():
        frame_size = stream.camera.resolution
        # path = str(Path(str(self.destination_path), "recordings", f"video_{port}.mp4"))
        path = str(Path(Path(__file__).parent, f"test_record{port}.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        fps = 10 # I believe that this will work....
        frame_size = stream.camera.resolution
            
        writer = cv2.VideoWriter(path, fourcc,fps, frame_size )
        video_writers[port] = writer

    while True:
        frame_bundle = bundle_in_q.get() 

        for port, bundle in frame_bundle.items():
            frame = bundle["frame"]
            frame_time = bundle["frame_time"]
            video_writers[port].write(frame)
        
            cv2.imshow(f"Cam {port}", frame)


        key = cv2.waitKey(1)

        if key == ord('q'):
            break

thread = Thread(target=frame_writer, args = [], daemon=False)
thread.start()
