
import pyxy3d.logger

import cv2
from pathlib import Path
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.interface import FramePacket
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.cameras.camera_array import CameraData
from pyxy3d.interface import Tracker
from pyxy3d.recording.video_recorder import VideoRecorder

logger = pyxy3d.logger.get(__name__)

class SynchronizedStreamManager():
    """
    This manager has substantial and broad responsibilities that include creation of:
    - streams
    - synchronizer
    - video recorder
    - frame dictionary emitter for GUI

    For a given recording directory, it manages the overhead of processing multiple 
    recorded streams concurrently
    - establishes frame dictionary emitter that can be used to display the tracking results to the user while being processed
    - option to save out processed results to a subfolder that contains:
        - mp4 files with tracked points
        - xy.csv file of tracked points from each camera source as well as the sync index

    Admittedly large in scope, it offloads substantial responsibilties from the controller layer.
    """
    
    def __init__(self, recording_dir:Path, all_camera_data:dict[CameraData], tracker:Tracker=None) -> None:
        self.recording_dir = recording_dir
        self.all_camera_data = all_camera_data
        self.tracker = tracker
        
        self.subfolder_name = "processed" if tracker is None else self.tracker.name
        self.output_dir = Path(self.recording_dir, self.subfolder_name)

        # To be filled when loading stream tools
        self.streams = {}
        self.frame_emitters = {}
        self.load_stream_tools()

    
    def load_stream_tools(self):
        
        for camera in self.all_camera_data.values():

            stream = RecordedStream(
            directory=self.recording_dir,
            port=camera.port,
            rotation_count=camera.rotation_count,
            tracker=self.tracker,
            break_on_last=True) 
        
            self.streams[camera.port] = stream
        
        self.synchronizer = Synchronizer(self.streams)    
        self.recorder = VideoRecorder(self.synchronizer,suffix= self.tracker.name )

    def process_streams(self, fps_target=None):
        self.recorder.start_recording(self.output_dir, include_video=True, show_points=True, store_point_history=True)
        
        for port, stream in self.streams.items():
            if fps_target is not None:
                stream.set_fps_target(fps_target)

            stream.play_video()
        
    