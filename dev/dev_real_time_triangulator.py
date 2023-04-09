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
import numpy as np
import numpy as np
import cv2
from typing import Dict, Tuple
import pandas as pd
from numba.typed import Dict

session_path = Path("tests", "sessions", "post_optimization")

config = Configurator(session_path)


charuco: Charuco = config.get_charuco()
camera_array: CameraArray = config.get_camera_array()

# logger.info(f"Creating RecordedStreamPool")
# stream_pool = RecordedStreamPool(session_path, charuco=charuco)
# logger.info("Creating Synchronizer")
# syncr = Synchronizer(stream_pool.streams, fps_target=None)
# stream_pool.play_videos()


#### Basic code for interfacing with in-progress RealTimeTriangulator
#### Just run off of saved point_data.csv for development/testing
# real_time_triangulator = RealTimeTriangulator(camera_array, syncr)
# while real_time_triangulator.running:
#     sleep(1)

# # packet 20 appears to be a good sample for development...
# sync_packet: SyncPacket = real_time_triangulator._sync_packet_history[20]

# point_packets = {}
# for port, packet in sync_packet.frame_packets.items():
#     point_packets[port] = packet.points

# save out point packets to csv for storage and refresh later

# %%
# Load point data
points = pd.read_csv(Path(session_path, "point_data.csv"))
#%%

camera_indices = points["port"].to_numpy()
sync_indices = points["sync_index"]
point_indices = points["point_id"]
img_x = points["img_loc_x"]
img_y = points["img_loc_y"]
img = np.vstack([img_x,img_y]).T
#%%
# img = 
# carryover from anipose/FMC
@jit(nopython=True, parallel=True)
def triangulate_simple(points, camera_ids, projection_matrices):
    num_cams = len(camera_ids)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        P = projection_matrices[camera_ids[i]]
        A[(i * 2) : (i * 2 + 1)] = x * P[2] - P[0]
        A[(i * 2 + 1) : (i * 2 + 2)] = y * P[2] - P[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d

####################### Function from ChatGPT Discussion #################################
# def triangulate_points_modified(point_packets: Dict[int, PointPacket], camera_data: Dict[int, CameraData]) -> Dict[int, np.ndarray]:
points_3d = {}
processed_point_ids = set()

projection_matrices = Dict()
for port, cam in camera_array.cameras.items():
    projection_matrices[port] = cam.projection_matrix

for sync_id in np.unique(sync_indices):
    # these arrays should be providing the basic structure of the PointPacket
    current_camera_indices = camera_indices[sync_indices==sync_id]
    current_point_id = point_indices[sync_indices==sync_id]
    current_img = img[sync_indices==sync_id]

    # only attempt to process points with multiple views
    # iterated across the current points to find those with multiple views
    unique_points, point_counts = np.unique(current_point_id, return_counts=True)
    for index in range(len(point_counts)):
        if point_counts[index] > 1:
            # triangulate that points...
            point = unique_points[index]
            points_xy = current_img[current_point_id==point]
            camera_ids = current_camera_indices[current_point_id==point]
            logger.info(f"Calculating xyz for point {point} at sync index {sync_id}")
            point_xyz = triangulate_simple(points_xy,camera_ids, projection_matrices)
# 
# for point_indices in np.unique(np.concatenate([point_packet.point_id for point_packet in point_packets.values()])):
#     points = []
#     camera_mats = []

#     # MP addition to CGPT code...
#     camera_data = camera_array.cameras

#     for cam_id, point_packet in point_packets.items():
#         if cam_id not in camera_data:
#             continue
#         logger.info(f"Doing something to camera {cam_id} and point id {point_indices}")
#         cam = camera_data[cam_id]

#         if cam.translation is None or cam.rotation is None:
#             continue

#         if point_indices in point_packet.point_id:
#             idx = np.where(point_packet.point_id == point_indices)[0][0]
#             img_point = point_packet.img_loc[idx]
#             # note I'm using squeeze here which I'm generally loathe to do
#             # but it is literally just an x,y coordinate
#             undistorted_point = cv2.undistortPoints(img_point, cam.matrix, cam.distortions, P=cam.matrix).squeeze()

#             # R = cam.rotation
#             RT = cam.transformation[0:3,:]
#             P = cam.matrix @ RT

#             points.append(undistorted_point)
#             camera_mats.append(P)

#     if len(points) >= 2:
#         p3d = triangulate_simple(np.array(points), np.array(camera_mats))
#         points_3d[point_indices] = p3d

# %%
