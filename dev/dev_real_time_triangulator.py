
""""
The Place where I'm putting together the RealTimeTriangulator working stuff that should one day become a test

Hopefully I can keep things clean enough for that...

"""
# %%
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)
from time import sleep
from dataclasses import asdict

from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.cameras.data_packets import PointPacket, FramePacket, SyncPacket
from pyxy3d.triangulate.real_time_triangulator import RealTimeTriangulator
from pyxy3d.cameras.camera_array import CameraArray, CameraData, get_camera_array
from pyxy3d.recording.recorded_stream import RecordedStreamPool
from pyxy3d.calibration.charuco import Charuco, get_charuco
from pyxy3d.configurator import Configurator

from pathlib import Path
from numba import jit
from numba.typed import Dict, List
import numpy as np
import cv2
import pandas as pd
from time import time
from pyxy3d import __root__

session_path = Path(__root__,"dev", "sample_sessions", "post_optimization")

config = Configurator(session_path)
origin_sync_index = config.dict["capture_volume"]["origin_sync_index"]

charuco: Charuco = config.get_charuco()
camera_array: CameraArray = config.get_camera_array()

logger.info(f"Creating RecordedStreamPool")
stream_pool = RecordedStreamPool(session_path, charuco=charuco)
logger.info("Creating Synchronizer")
syncr = Synchronizer(stream_pool.streams, fps_target=None)


#### Basic code for interfacing with in-progress RealTimeTriangulator
#### Just run off of saved point_data.csv for development/testing
real_time_triangulator = RealTimeTriangulator(camera_array, syncr, output_directory=session_path)
stream_pool.play_videos()
while real_time_triangulator.running:
    sleep(1)

#%%