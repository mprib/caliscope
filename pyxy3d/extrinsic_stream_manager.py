
import pyxy3d.logger

import cv2
from pathlib import Path
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.interface import FramePacket
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.cameras.camera_array import CameraData

logger = pyxy3d.logger.get(__name__)

class ExtrinsicStreamManager():
    """
    
    """
    
    def __init__(self, extrinsic_dir:Path) -> None:
    
        self.extrinsic_dir = extrinsic_dir

        self.streams = {}
        self.frame_emitters = {}

    def add_stream(self,camera:CameraData):
        
        stream = RecordedStream(
        directory=self.extrinsic_dir,
        port=camera.port,
        rotation_count=camera.rotation_count,
        tracker=self.charuco_tracker,
        break_on_last=False) 

        
        self.streams[camera.port] = stream


    def load_synchronizer(self):
        """
        cannot load synchronizer until all camera streams are loaded
        """
        pass