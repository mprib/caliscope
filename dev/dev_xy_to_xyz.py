"""
Building test regarding the conversion of the xy.csv datafile into an xyz.csv datafile.
I suppose that it may make sense to run this through the same processing pipeline 
as the SyncPacketTriangulator...no sense reinventing the wheel.



"""

# %%

import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)
import time
import pandas as pd
from pathlib import Path
from numba.typed import Dict, List
import numpy as np

from pyxy3d.helper import copy_contents
from pyxy3d import __root__
from pyxy3d.triangulate.sync_packet_triangulator import triangulate_sync_index
from pyxy3d.configurator import Configurator
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.post_processing_pipelines import create_xy_points
from pyxy3d.trackers.hand_tracker import HandTrackerFactory

# load in file of xy point data
origin_data = Path(__root__, "tests", "sessions", "4_cam_recording")
working_data = Path(__root__, "tests", "sessions_copy_delete", "4_cam_recording")

copy_contents(origin_data, working_data)

config = Configurator(working_data)
recording_directory = Path(working_data, "recording_1_frames_processed")
xy_path = Path(recording_directory, "xy.csv")


# need to initially create the xy data....
# create_xy_points(
#     config=config,
#     recording_directory=recording_directory,
#     tracker_factory=HandTrackerFactory(),
# )

xy_data = pd.read_csv(xy_path)
camera_array = config.get_camera_array()


def triangulate_xy_data(xy_data:pd.DataFrame, camera_array:CameraArray)->Dict[str,List]:

    # assemble numba compatible dictionary
    projection_matrices = Dict()
    for port, cam in camera_array.cameras.items():
        projection_matrices[int(port)] = cam.projection_matrix
    
    xyz_history = {"point_id":[],
                   "x_coord": [],
                   "y_coord": [],
                   "z_coord": [],}
    
    for index in xy_data["sync_index"].unique():
        
        active_index = xy_data["sync_index"] == index
        cameras = xy_data["port"][active_index].to_numpy()
        point_ids = xy_data["point_id"][active_index].to_numpy()
        img_loc_x = xy_data["img_loc_x"][active_index].to_numpy()
        img_loc_y = xy_data["img_loc_y"][active_index].to_numpy()
        imgs_xy = np.vstack([img_loc_x, img_loc_y]).T

        point_id_xyz, points_xyz = triangulate_sync_index(
            projection_matrices, cameras, point_ids, imgs_xy
        )

        if len(point_id_xyz) > 0:        
            # there are points to store so store them...
            points_xyz = np.array(points_xyz)
            xyz_history["point_id"].extend(point_id_xyz)
            xyz_history["x_coord"].extend(points_xyz[:,0].tolist())
            xyz_history["y_coord"].extend(points_xyz[:,1].tolist())
            xyz_history["z_coord"].extend(points_xyz[:,2].tolist())

    return xyz_history

start = time.time()
logger.info(f"beginning triangulation at {time.time()}")
xyz_history = triangulate_xy_data(xy_data, camera_array)
logger.info(f"ending triangulation at {time.time()}")
stop = time.time()
logger.info(f"Elapsed time is {stop-start}")