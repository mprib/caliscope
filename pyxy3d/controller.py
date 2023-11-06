
from typing import Optional
from PySide6.QtCore import QObject

from threading import Thread, Event
from time import sleep
from pathlib import Path
from datetime import datetime
from os.path import exists
import numpy as np
import toml
from dataclasses import asdict
import  cv2
from concurrent.futures import ThreadPoolExecutor
from queue import Queue 

import pyxy3d.logger
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.cameras.camera import Camera
from pyxy3d.cameras.camera_array import CameraArray, CameraData
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.configurator import Configurator
from pyxy3d.playback_frame_emitter import PlaybackFrameEmitter

logger = pyxy3d.logger.get(__name__)


class Controller(QObject):
    """
    Thin layer to integrate GUI and backend 
    """ 

    def __init__(self, workspace_dir:Path):
        super().__init__()
        self.workspace = workspace_dir
        self.config = Configurator(self.workspace)
    
        # set up cameras via intrinsic calibration
        self.cameras = {}
        # streams will be used to play back recorded video with tracked markers to select frames
        self.streams = {}
        self.frame_emitters = {}

    def add_camera(self, source_path:Path=None)->int:
        """
        When adding source video to calibrate a camera, the function returns the camera index
        """
        index = len(self.streams)
        self.streams[index] = RecordedStream(source_file=source_path)
        self.frame_emitters[index] = PlaybackFrameEmitter()

        frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    
    def pause_stream(self,index):
        self.streams[index].pause()
            
    def unpause_stream(self,index):
        self.streams[index].unpause()

    def stream_jump_frame(self,index, frame_index):
        self.streams[index].jump_frame(frame_index)
     
        
class RecordedStream():
    def __init__(self, source_file:Path) -> None:
        self.source_file = source_file
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.jump_frame = Queue()
        self.play_blocker = Queue()

    def subscribe(self, q:Queue):
        self.out_q = q

    def jump_frame(self, frame_index:int):
        self.jump_q.put(frame_index)
   
    def pause(self):
        _ = self.play_blocker.get()
        
    def unpause(self):
        self.play_blocker.put(True)
         
    def play(self):
        self.playing.set()
        def play_worker():     
            while True:
                
                _ = self.play_blocker.get() # will hold if self.pause has removed item from queue

                success,frame = self.cap.read()
                self.out_q.put(frame)
                sleep(1/self.frame_rate)

                self.play_blocker.put(True) # keep loop active
                
                    
        self.play_thread = Thread(target=play_worker, args=[], daemon=True)
        self.play_thread.start()
            