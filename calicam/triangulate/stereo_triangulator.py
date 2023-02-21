# this class is only a way to hold data related to the stereocamera triangulation.
# These will load from a config file (.toml) and provide a way for the 3D triangulation
# and plotting to manage the parameters. It feels like some duplication of the camera object,
# but I want something that is designed to be simple and not actually manage the cameras, just
# organize the saved data

import calicam.logger
logger = calicam.logger.get(__name__)

from queue import Queue
from threading import Thread, Event
from dataclasses import dataclass
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from calicam.triangulate.paired_point_builder import PairedPointBuilder, PairedPointsPacket
from calicam.cameras.camera_array import CameraData

class StereoTriangulator:
    def __init__(self, camera_A: CameraData, camera_B: CameraData):

        self.camera_A = camera_A
        self.camera_B = camera_B
        self.portA = camera_A.port
        self.portB = camera_B.port
        self.pair = (self.portA, self.portB)

        self.build_projection_matrices()

    def build_projection_matrices(self):

        # Camera parameters are position in a world frame of reference
        # which is synonymous to the anchor camera frame prior to setting origin
        # Projection matrix is to re-orient a point from the world position
        # to a camera frame of reference, therefore is inverted (rotation)/negated (translation)
        # I believe this is the correct interpretation and appears to yield
        # reasonable results
        rot_A = np.linalg.inv(self.camera_A.rotation)
        trans_A = np.array(self.camera_A.translation) * -1
        rot_trans_A = np.concatenate([rot_A, trans_A], axis=-1)
        mtx_A = self.camera_A.camera_matrix
        self.proj_A = mtx_A @ rot_trans_A  # projection matrix for CamA

        rot_B = np.linalg.inv(self.camera_B.rotation)
        trans_B = np.array(self.camera_B.translation) * -1
        rot_trans_B = np.concatenate([rot_B, trans_B], axis=-1)
        mtx_B = self.camera_B.camera_matrix
        self.proj_B = mtx_B @ rot_trans_B  # projection matrix for CamB

    def get_3D_points(self, paired_points:PairedPointsPacket):
            
        if len(paired_points.common_ids) > 0:
            points_A_raw = paired_points.img_loc_A
            points_B_raw = paired_points.img_loc_B

            points_A_undistorted = self.undistort(points_A_raw,self.camera_A)
            points_B_undistorted = self.undistort(points_B_raw,self.camera_B)

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

        triangulated_points = TriangulatedPointsPacket(
            sync_index=paired_points.sync_index,
            pair=self.pair,
            point_ids=paired_points.common_ids,
            xyz=xyz,
            xy_A_raw=points_A_raw,
            xy_B_raw=points_B_raw,
            xy_A_undistorted=points_A_undistorted,
            xy_B_undistorted=points_B_undistorted,
        )


        return triangulated_points

    def undistort(self, points, camera: CameraData, iter_num=3):
        # implementing a function described here: https://yangyushi.github.io/code/2020/03/04/opencv-undistort.html
        # supposedly a better implementation than OpenCV
        k1, k2, p1, p2, k3 = camera.distortion[0]
        fx, fy = camera.camera_matrix[0, 0], camera.camera_matrix[1, 1]
        cx, cy = camera.camera_matrix[:2, 2]
        
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


@dataclass
class TriangulatedPointsPacket:
    pair: tuple  # parent pair
    point_ids: np.ndarray
    xyz: np.ndarray
    sync_index: int
    xy_A_raw: np.ndarray
    xy_B_raw: np.ndarray
    xy_A_undistorted: np.ndarray
    xy_B_undistorted: np.ndarray

    def to_dict(self):
        num_rows = len(self.point_ids)

        if num_rows == 0:
            return None 
        else:
            pair_list = [self.pair] * num_rows
            # time_list = [self.time] * num_rows
            sync_index_list = [self.sync_index] * num_rows
            id_list = self.point_ids.tolist()
            port_A_list = [self.pair[0]] * num_rows
            port_B_list = [self.pair[1]] * num_rows
            x_list = self.xyz[:, 0].tolist()
            y_list = self.xyz[:, 1].tolist()
            z_list = self.xyz[:, 2].tolist()
            x_A_raw = self.xy_A_raw.T[0,:].tolist()
            y_A_raw = self.xy_A_raw.T[1,:].tolist()
            x_B_raw = self.xy_B_raw.T[0,: ].tolist()
            y_B_raw = self.xy_B_raw.T[1,: ].tolist()
            x_A_undistorted = self.xy_A_undistorted[0,:].tolist()
            y_A_undistorted = self.xy_A_undistorted[1,:].tolist()
            x_B_undistorted = self.xy_B_undistorted[0,: ].tolist()
            y_B_undistorted = self.xy_B_undistorted[1,: ].tolist()

            data = {
                "pair": pair_list,
                "port_A": port_A_list,
                "port_B": port_B_list,
                # "time": time_list,
                "sync_index": sync_index_list,
                "id": id_list,
                "x_pos": x_list,
                "y_pos": y_list,
                "z_pos": z_list,
                "x_A_raw": x_A_raw,
                "y_A_raw": y_A_raw,
                "x_B_raw": x_B_raw,
                "y_B_raw": y_B_raw,
                "x_A_undistorted": x_A_undistorted,
                "y_A_undistorted": y_A_undistorted,
                "x_B_undistorted": x_B_undistorted,
                "y_B_undistorted": y_B_undistorted,
            }

            return data


if __name__ == "__main__":

    from calicam.recording.recorded_stream import RecordedStreamPool
    from calicam.cameras.synchronizer import Synchronizer
    from calicam.calibration.charuco import Charuco
    from calicam.calibration.corner_tracker import CornerTracker
    from calicam.cameras.camera_array import CameraArrayBuilder, CameraArray, CameraData

    repo = Path(str(Path(__file__)).split("calicam")[0],"calicam")

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
    point_stream = PairedPointBuilder(
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
