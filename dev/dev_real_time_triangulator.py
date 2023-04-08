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


####################### Function from ChatGPT Discussion #################################
# def triangulate_points(point_packets: Dict[int, PointPacket], camera_data: Dict[int, CameraData]) -> Dict[int, np.ndarray]:
points_3d = {}
processed_point_ids = set()

for cam_id, point_packet in point_packets.items():
    if cam_id not in camera_array.cameras:
        continue

    cam = camera_array.cameras[cam_id]


    for idx, point_id in enumerate(point_packet.point_id):
        if point_id in processed_point_ids:
            continue

        img_point = point_packet.img_loc[idx]
        undistorted_point = cv2.undistortPoints(img_point, cam.matrix, cam.distortions, P=cam.matrix)

        R = cv2.Rodrigues(cam.rotation)[0]  # Convert rotation vector to rotation matrix
        RT = np.hstack((R, cam.translation))

        # Check for another camera observing the same point
        for other_cam_id, other_point_packet in point_packets.items():
            if other_cam_id == cam_id or point_id not in other_point_packet.point_id:
                continue

            other_cam = camera_array.cameras[other_cam_id]

            if other_cam.translation is None or other_cam.rotation is None:
                continue

            other_idx = np.where(other_point_packet.point_id == point_id)[0][0]
            other_img_point = other_point_packet.img_loc[other_idx]
            other_undistorted_point = cv2.undistortPoints(other_img_point, other_cam.matrix, other_cam.distortions, P=other_cam.matrix)

            other_R = cv2.Rodrigues(other_cam.rotation)[0]
            other_RT = np.hstack((other_R, other_cam.translation))

            # Triangulate the point using both camera poses
            points_homogeneous = cv2.triangulatePoints(RT, other_RT, undistorted_point, other_undistorted_point)
            points_3d[point_id] = (points_homogeneous / points_homogeneous[3])[:3].reshape(-1)

            processed_point_ids.add(point_id)
            break

points_3d
# %%
