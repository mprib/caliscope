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

# Load point data
points_xy = pd.read_csv(Path(session_path, "point_data.csv"))

camera_indices = points_xy["port"].to_numpy()
sync_indices = points_xy["sync_index"].to_numpy()
point_indices = points_xy["point_id"].to_numpy()
img_x = points_xy["img_loc_x"].to_numpy()
img_y = points_xy["img_loc_y"].to_numpy()
img = np.vstack([img_x, img_y]).T


# @jit(nopython=True, parallel=True)
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

# @jit(nopython=True)
def unique_with_counts(arr):
    sorted_arr = np.sort(arr)
    unique_values = [sorted_arr[0]]
    counts = [1]

    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] != sorted_arr[i - 1]:
            unique_values.append(sorted_arr[i])
            counts.append(1)
        else:
            counts[-1] += 1

    return np.array(unique_values), np.array(counts)
####################### Function from ChatGPT Discussion #################################
# def triangulate_points_modified(point_packets: Dict[int, PointPacket], camera_data: Dict[int, CameraData]) -> Dict[int, np.ndarray]:


# @jit(nopython=True, parallel=True, cache=True)
def triangulate_sync_index(
    projection_matrices, current_camera_indices, current_point_id, current_img
):
    # sync_indices_xyz = List()
    point_indices_xyz = List()
    obj_xyz = List()

    unique_points, point_counts = unique_with_counts(current_point_id)
    for index in range(len(point_counts)):
        if point_counts[index] > 1:
            # triangulate that points...
            point = unique_points[index]
            points_xy = current_img[current_point_id == point]
            camera_ids = current_camera_indices[current_point_id == point]
            # logger.info(f"Calculating xyz for point {point} at sync index {sync_id}")
            # point_xyz = triangulate_simple(points_xy, camera_ids, projection_matrices)

            num_cams = len(camera_ids)
            A = np.zeros((num_cams * 2, 4))
            for i in range(num_cams):
                x, y = points_xy[i]
                P = projection_matrices[camera_ids[i]]
                A[(i * 2) : (i * 2 + 1)] = x * P[2] - P[0]
                A[(i * 2 + 1) : (i * 2 + 2)] = y * P[2] - P[1]
            u, s, vh = np.linalg.svd(A, full_matrices=True)
            point_xyzw = vh[-1]
            point_xyz = point_xyzw[:3] / point_xyzw[3]

            # sync_indices_xyz.append(sync_id)
            point_indices_xyz.append(point)
            obj_xyz.append(point_xyz)

    return point_indices_xyz, obj_xyz


# initialize numba specific datastructures that will be used to access data within function

projection_matrices = Dict()
for port, cam in camera_array.cameras.items():
    projection_matrices[port] = cam.projection_matrix

all_sync_indices_xyz = List()
all_point_indices_xyz = List()
all_obj_xyz = List()

logger.info(f"Begin processing sync indices")
for sync_id in np.unique(sync_indices):
    # these arrays should be providing the basic structure of the PointPacket
    current_camera_indices = camera_indices[sync_indices == sync_id]
    current_point_id = point_indices[sync_indices == sync_id]
    current_img = img[sync_indices == sync_id]

    start_sync_index = time()
    # only attempt to process points with multiple views
    # iterated across the current points to find those with multiple views

    point_indices_xyz, obj_xyz = triangulate_sync_index(
        projection_matrices, current_camera_indices, current_point_id, current_img
    )
    stop_sync_index = time()
    current_sync_indices = List([sync_id]*len(point_indices_xyz))
    all_sync_indices_xyz.extend(current_sync_indices)
    all_point_indices_xyz.extend(point_indices_xyz)
    all_obj_xyz.extend(obj_xyz)

    # logger.info(f"Finished processing sync id {sync_id} at {stop_sync_index}")
    elapsed_time = start_sync_index - stop_sync_index
    logger.info(f"Elapsed time to process sync index {sync_id} is {elapsed_time}")

    # only attempt to process points with multiple views

# %%
