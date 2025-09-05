from itertools import combinations

import cv2
import numpy as np

import caliscope.logger
from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.triangulate.stereo_points_builder import StereoPointsPacket, SyncedStereoPointsPacket

logger = caliscope.logger.get(__name__)


class StereoTriangulator:
    def __init__(self, camera_array: CameraArray):
        self.camera_array = camera_array

        # pull ports list from camera_array.port_index
        # to ensure only non-ignored cameras are processed
        self.ports = list(camera_array.posed_port_to_index.keys())
        self.pairs = [(i, j) for i, j in combinations(self.ports, 2) if i < j]

        # create the triangulators for each pair of cameras
        self.triangulators = {}
        for pair in self.pairs:
            port_A = pair[0]
            port_B = pair[1]

            camera_A: CameraData = self.camera_array.cameras[port_A]
            camera_B: CameraData = self.camera_array.cameras[port_B]

            self.triangulators[pair] = StereoPairTriangulator(camera_A, camera_B)

    def triangulate_synced_points(self, synced_paired_points: SyncedStereoPointsPacket):
        for pair, paired_point_packet in synced_paired_points.stereo_points_packets.items():
            if paired_point_packet is not None:
                self.triangulators[pair].add_3D_points(paired_point_packet)


class StereoPairTriangulator:
    def __init__(self, camera_A: CameraData, camera_B: CameraData):
        self.camera_A = camera_A
        self.camera_B = camera_B
        self.portA = camera_A.port
        self.portB = camera_B.port
        self.pair = (self.portA, self.portB)

    def add_3D_points(self, paired_points: StereoPointsPacket):
        if len(paired_points.common_ids) > 0:
            xy_A = paired_points.img_loc_A
            xy_B = paired_points.img_loc_B

        if xy_A.shape[0] > 0:
            logger.info(f"Triangulating points in common between ports {self.camera_A.port} and {self.camera_B.port}")
            points_A_undistorted = self.camera_A.undistort_points(xy_A).T
            points_B_undistorted = self.camera_B.undistort_points(xy_B).T

            # triangulate joints outputs data in 4D homogenous coordinate system
            # note that these are in a world frame of reference
            xyzw_h = cv2.triangulatePoints(
                self.camera_A.normalized_projection_matrix,
                self.camera_B.normalized_projection_matrix,
                points_A_undistorted,
                points_B_undistorted,
            )

            xyz_h = xyzw_h.T[:, :3]
            w = xyzw_h[3, :]
            xyz = np.divide(xyz_h.T, w).T  # convert to euclidean coordinates
        else:
            xyz = np.array([])

        # update the paired point packet with the 3d positions
        paired_points.xyz = xyz
