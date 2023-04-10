
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
# origin_sync_index = config.dict["capture_volume"]["origin_sync_index"]

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
# need to compare the output of the triangulator to the point_estimats
# this is nice because it's two totally different processing pipelines
# but sync indices will be different, so just compare mean positions
# which should be quite close

xyz_history = pd.read_csv(Path(session_path,"xyz_history.csv"))
xyz_config = np.array(config.dict["point_estimates"]["obj"])
triangulator_x_mean = xyz_history["x_coord"].mean()
triangulator_y_mean = xyz_history["y_coord"].mean()
triangulator_z_mean = xyz_history["z_coord"].mean()

config_x_mean = xyz_config[:,0].mean()
config_y_mean = xyz_config[:,1].mean()
config_z_mean = xyz_config[:,2].mean()

logger.info(f"x: {round(triangulator_x_mean,4)} vs {round(config_x_mean,4)} ")
logger.info(f"y: {round(triangulator_y_mean,4)} vs {round(config_y_mean,4)} ")
logger.info(f"z: {round(triangulator_z_mean,4)} vs {round(config_z_mean,4)} ")
# at this point, the data should be saved out...
# %%
