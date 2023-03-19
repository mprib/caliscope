# this class is only a way to hold data related to the stereocamera triangulation.
# These will load from a config file (.toml) and provide a way for the 3D triangulation
# and plotting to manage the parameters. It feels like some duplication of the camera object,
# but I want something that is designed to be simple and not actually manage the cameras, just
# organize the saved data

import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from queue import Queue
from threading import Thread, Event
from dataclasses import dataclass
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

from pyxy3d.triangulate.stereo_points_builder import StereoPointsBuilder, StereoPointsPacket
from pyxy3d.cameras.data_packets import PointPacket, FramePacket, SyncPacket
from pyxy3d.triangulate.stereo_points_builder import StereoPointsPacket, SynchedStereoPointsPacket


from pyxy3d.cameras.camera_array import CameraData, CameraArray


class ArrayTriangulator:
    
    def __init__(self, camera_array: CameraArray):
        self.camera_array = camera_array
        
        self.ports = list(camera_array.cameras.keys())
        self.pairs = [(i,j) for i,j in combinations(self.ports,2) if i<j]


        # create the triangulators for each pair of cameras
        self.triangulators = {}
        for pair in self.pairs:
            port_A = pair[0]
            port_B = pair[1]
            
            camera_A:CameraData = self.camera_array.cameras[port_A]
            camera_B:CameraData = self.camera_array.cameras[port_B]

            self.triangulators[pair] = StereoTriangulator(camera_A, camera_B)
            
    def triangulate_synched_points(self, synced_paired_points:SynchedStereoPointsPacket):
        for pair,paired_point_packet  in synced_paired_points.stereo_points_packets.items():
            if paired_point_packet is not None:
                self.triangulators[pair].add_3D_points(paired_point_packet)



class StereoTriangulator:
    def __init__(self, camera_A: CameraData, camera_B: CameraData):

        self.camera_A = camera_A
        self.camera_B = camera_B
        self.portA = camera_A.port
        self.portB = camera_B.port
        self.pair = (self.portA, self.portB)

        self.build_projection_matrices()

    def build_projection_matrices(self):

        # attempting to create something that integrates with the new set_cameras_refactor
        # rot_A = np.linalg.inv(self.camera_A.rotation)
        # trans_A = np.array(self.camera_A.translation) * -1
        rot_trans_A = np.column_stack([self.camera_A.rotation, self.camera_A.translation])
        mtx_A = self.camera_A.matrix
        self.proj_A = mtx_A @ rot_trans_A  # projection matrix for CamA

        # rot_B = np.linalg.inv(self.camera_B.rotation)
        # trans_B = np.array(self.camera_B.translation) * -1
        rot_trans_B = np.column_stack([self.camera_B.rotation, self.camera_B.translation])
        mtx_B = self.camera_B.matrix
        self.proj_B = mtx_B @ rot_trans_B  # projection matrix for CamB

    def build_projection_matrices_old(self):

        # inversion/negation of R t here is legacy code that  
        # was based on my understanding at the time of frames of reference.
        # and it yields highly reasonable results. 
        # see https://stackoverflow.com/questions/17210424/3d-camera-coordinates-to-world-coordinates-change-of-basis
        # for a potential explanation. 
        # I would expect this to be a more common topic for computer vision forums
        # but I can't really find a reference to this and it bothers me
        rot_A = np.linalg.inv(self.camera_A.rotation)
        trans_A = np.array(self.camera_A.translation) * -1
        rot_trans_A = np.column_stack([rot_A, trans_A])
        mtx_A = self.camera_A.matrix
        self.proj_A = mtx_A @ rot_trans_A  # projection matrix for CamA

        rot_B = np.linalg.inv(self.camera_B.rotation)
        trans_B = np.array(self.camera_B.translation) * -1
        rot_trans_B = np.column_stack([rot_B, trans_B])
        mtx_B = self.camera_B.matrix
        self.proj_B = mtx_B @ rot_trans_B  # projection matrix for CamB

    def add_3D_points(self, paired_points:StereoPointsPacket):
            
        if len(paired_points.common_ids) > 0:
            xy_A = paired_points.img_loc_A
            xy_B = paired_points.img_loc_B
         
        if xy_A.shape[0] > 0:

            points_A_undistorted = self.undistort(xy_A,self.camera_A)
            points_B_undistorted = self.undistort(xy_B,self.camera_B)

            # triangulate points outputs data in 4D homogenous coordinate system
            # note that these are in a world frame of reference
            xyzw_h = cv2.triangulatePoints(
                self.proj_A, self.proj_B, points_A_undistorted, points_B_undistorted
            )

            xyz_h = xyzw_h.T[:, :3]
            w = xyzw_h[3, :]
            xyz = np.divide(xyz_h.T, w).T  # convert to euclidean coordinates
        else:
            xyz = np.array([])

        # update the paired point packet with the 3d positions
        paired_points.xyz = xyz
        
        
    def undistort(self, points, camera: CameraData, iter_num=3):
        # implementing a function described here: https://yangyushi.github.io/code/2020/03/04/opencv-undistort.html
        # supposedly a better implementation than OpenCV
        k1, k2, p1, p2, k3 = camera.distortions
        fx, fy = camera.matrix[0, 0], camera.matrix[1, 1]
        cx, cy = camera.matrix[:2, 2]
        
        # note I just made an edit to transpose these...I believe this is a consequence
        # of the recent switch to the PointPacket in the processing pipeline
        x, y = points.T[0], points.T[1]
        # x, y = float(point[0]), float(point[1])

        x = (x - cx) / fx
        x0 = x
        y = (y - cy) / fy
        y0 = y

        for _ in range(iter_num):
            r2 = x**2 + y**2
            k_inv = 1 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
            delta_x = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
            delta_y = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
            x = (x0 - delta_x) * k_inv
            y = (y0 - delta_y) * k_inv
        return np.array((x * fx + cx, y * fy + cy))




if __name__ == "__main__":

    from pyxy3d.recording.recorded_stream import RecordedStreamPool
    from pyxy3d.cameras.synchronizer import Synchronizer
    from pyxy3d.calibration.charuco import Charuco
    from pyxy3d.calibration.corner_tracker import CornerTracker
    from pyxy3d.cameras.camera_array import CameraArrayBuilder, CameraArray, CameraData

    repo = Path(str(Path(__file__)).split("pyxy")[0],"pyxy")

    config_path = Path(repo, "sessions", "iterative_adjustment", "config.toml")
    camera_array = CameraArrayBuilder(config_path).get_camera_array()

    # create playback streams to provide to synchronizer
    recorded_data = Path(repo, "sessions", "iterative_adjustment", "recording")
    ports = [0, 1]
    recorded_stream_pool = RecordedStreamPool(ports, recorded_data)
    syncr = Synchronizer(
        recorded_stream_pool.streams, fps_target=None
    )  # no fps target b/c not playing back for visual display
    recorded_stream_pool.play_videos()

    # create a corner tracker to locate board corners
    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )
    trackr = CornerTracker(charuco)

    # create a commmon point finder to grab charuco corners shared between the pair of ports
    pairs = [(0, 1)]
    point_stream = StereoPointsBuilder(
        synchronizer=syncr,
        pairs=pairs,
        tracker=trackr,
    )

    camA, camB = camera_array.cameras[0], camera_array.cameras[2]
    pair = (camA.port, camB.port)

    # test_pair_out_q = Queue(-1)
    triangulatr = StereoTriangulator(camA, camB)
    frames_processed = 0

    while True:
        paired_points = point_stream.out_q.get()
        if paired_points.pair == (0, 1):
            triangulatr.in_q.put(paired_points)

        # print(all_pairs_common_points)
        # pair_points = all_pairs_common_points[pair]
        # if pair_points is not None:
        # triangulatr.in_q.put(paired_points)
        packet_3d = triangulatr.out_q.get()
        print(packet_3d.to_dict())
        frames_processed += 1
        # print(f"Frames Processed: {frames_processed}")
        # print(f"Time: {packet_3d.time}")
