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



session_path = Path("tests", "sessions", "post_optimization")

config = Configurator(session_path)


charuco: Charuco = config.get_charuco()
camera_array: CameraArray = config.get_camera_array()

logger.info(f"Creating RecordedStreamPool")
stream_pool = RecordedStreamPool(session_path, charuco=charuco)
logger.info("Creating Synchronizer")
syncr = Synchronizer(stream_pool.streams, fps_target=None)
stream_pool.play_videos()

real_time_triangulator = RealTimeTriangulator(camera_array, syncr)

while real_time_triangulator.running:
    sleep(1)

# packet 20 appears to be a good sample for development...
sync_packet: SyncPacket = real_time_triangulator._sync_packet_history[20]

point_packets = {}
for port, packet in sync_packet.frame_packets.items():
    point_packets[port] = packet.points
# %%
# carryover from anipose/FMC
@jit(nopython=True, parallel=True)
def triangulate_simple(points, camera_mats):
    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2) : (i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1) : (i * 2 + 2)] = y * mat[2] - mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d

####################### Function from ChatGPT Discussion #################################
# def triangulate_points_modified(point_packets: Dict[int, PointPacket], camera_data: Dict[int, CameraData]) -> Dict[int, np.ndarray]:
points_3d = {}
processed_point_ids = set()

for point_id in np.unique(np.concatenate([point_packet.point_id for point_packet in point_packets.values()])):
    points = []
    camera_mats = []

    # MP addition to CGPT code...
    camera_data = camera_array.cameras

    for cam_id, point_packet in point_packets.items():
        if cam_id not in camera_data:
            continue
        logger.info(f"Doing something to camera {cam_id} and point id {point_id}")
        cam = camera_data[cam_id]

        if cam.translation is None or cam.rotation is None:
            continue

        if point_id in point_packet.point_id:
            idx = np.where(point_packet.point_id == point_id)[0][0]
            img_point = point_packet.img_loc[idx]
            # note I'm using squeeze here which I'm generally loathe to do
            # but it is literally just an x,y coordinate
            undistorted_point = cv2.undistortPoints(img_point, cam.matrix, cam.distortions, P=cam.matrix).squeeze()

            # R = cam.rotation
            RT = cam.transformation[0:3,:]
            P = cam.matrix @ RT

            points.append(undistorted_point)
            camera_mats.append(P)

    if len(points) >= 2:
        p3d = triangulate_simple(np.array(points), np.array(camera_mats))
        points_3d[point_id] = p3d

# %%
