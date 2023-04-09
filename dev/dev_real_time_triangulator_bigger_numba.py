## Just testing out an alternate setup of this to see what happens if I put
# an entire sync index processing in one numba function.


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

session_path = Path("tests", "sessions", "post_optimization")

config = Configurator(session_path)


charuco: Charuco = config.get_charuco()
camera_array: CameraArray = config.get_camera_array()

logger.info(f"Creating RecordedStreamPool")
stream_pool = RecordedStreamPool(session_path, charuco=charuco)
logger.info("Creating Synchronizer")
syncr = Synchronizer(stream_pool.streams, fps_target=None)
stream_pool.play_videos()


#### Basic code for interfacing with in-progress RealTimeTriangulator
#### Just run off of saved point_data.csv for development/testing
real_time_triangulator = RealTimeTriangulator(camera_array, syncr)
while real_time_triangulator.running:
    sleep(1)

# # packet 20 appears to be a good sample for development...
# sync_packet: SyncPacket = real_time_triangulator._sync_packet_history[20]

# point_packets = {}
# for port, packet in sync_packet.frame_packets.items():
#     point_packets[port] = packet.points

# save out point packets to csv for storage and refresh later

# Load point data
# points_xy = pd.read_csv(Path(session_path, "point_data.csv"))

# camera_indices = points_xy["port"].to_numpy()
# sync_indices = points_xy["sync_index"].to_numpy()
# point_indices = points_xy["point_id"].to_numpy()
# img_x = points_xy["img_loc_x"].to_numpy()
# img_y = points_xy["img_loc_y"].to_numpy()
# img = np.vstack([img_x, img_y]).T


# ####################### Function from ChatGPT Discussion #################################
# # def triangulate_points_modified(point_packets: Dict[int, PointPacket], camera_data: Dict[int, CameraData]) -> Dict[int, np.ndarray]:




# # initialize numba specific datastructures that will be used to access data within function

# projection_matrices = Dict()
# # projection_matrices = {}
# for port, cam in camera_array.cameras.items():
#     projection_matrices[port] = cam.projection_matrix

# # all_sync_indices_xyz = List()
# # all_point_indices_xyz = List()
# # all_obj_xyz = List()

# all_sync_id_xyz = []
# all_point_id_xyz = []
# all_points_xyz = []

# logger.info(f"Begin processing sync indices")
# for sync_index in np.unique(sync_indices):
#     # these arrays should be providing the basic structure of the PointPacket
#     cameras_at_sync_index = camera_indices[sync_indices == sync_index]
#     points_at_sync_index = point_indices[sync_indices == sync_index]
#     img_xy = img[sync_indices == sync_index]

#     start_sync_index = time()
#     # only attempt to process points with multiple views
#     # iterated across the current points to find those with multiple views

#     point_id_xyz, points_xyz = triangulate_sync_index(
#         projection_matrices, cameras_at_sync_index, points_at_sync_index, img_xy
#     )
#     stop_sync_index = time()
#     current_sync_index_list = [sync_index]*len(point_id_xyz)
#     all_sync_id_xyz.extend(current_sync_index_list)
#     all_point_id_xyz.extend(point_id_xyz)
#     all_points_xyz.extend(points_xyz)

#     # logger.info(f"Finished processing sync id {sync_id} at {stop_sync_index}")
#     elapsed_time = start_sync_index - stop_sync_index
#     logger.info(f"Elapsed time to process sync index {sync_index} is {elapsed_time}")

#     # only attempt to process points with multiple views

# %%

