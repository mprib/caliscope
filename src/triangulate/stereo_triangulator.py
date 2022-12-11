# this class is only a way to hold data related to the stereocamera triangulation.
# These will load from a config file (.toml) and provide a way for the 3D triangulation
# and plotting to manage the parameters. It feels like some duplication of the camera object,
# but I want something that is designed to be simple and not actually manage the cameras, just
# organize the saved data

import logging

LOG_FILE = r"log\stereo_triangulator.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO

LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from dataclasses import dataclass
import math
from os.path import exists
import toml
import numpy as np
import pandas as pd
import scipy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class CameraData:
    port: int
    resolution: tuple
    camera_matrix: np.ndarray
    error: float

    def __post_init__(self):
        # self.mesh = CameraMesh(self.resolution, self.camera_matrix).mesh
        # initialize to origin
        self.translation = np.array([0, 0, 0])
        self.rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


class StereoTriangulator:
    # created from a config.toml file, points within each camera frame can be provided to it
    # via self.get_3D_points(CommonPoints: pd.DataFrame)

    def __init__(self, portA, portB, config_path):
        self.portA = portA
        self.portB = portB
        self.config_path = config_path

        self.load_cams_from_config()
        self.build_projection_matrices()

    def load_cams_from_config(self):
        with open(self.config_path, "r") as f:
            logging.info(f"Loading config data located at: {self.config_path}")
            self.config = toml.load(self.config_path)

        self.camera_A = self.get_camera_at_origin(0)
        self.camera_B = self.get_camera_at_origin(1)

        # express location of camera B relative to Camera A
        rot, trans = self.get_extrinsic_params()
        self.camera_B.rotation = rot
        self.camera_B.translation = trans  # may come in with extra dims

    def get_camera_at_origin(self, port):

        data = self.config[f"cam_{port}"]

        resolution = tuple(data["resolution"])
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        error = data["error"]

        cam_data = CameraData(port, resolution, camera_matrix, error)
        logging.info(f"Loading camera data at port {port}: {str(cam_data)}")

        return cam_data

    def get_extrinsic_params(self):

        data = self.config[f"stereo_{self.portA}_{self.portB}"]
        rotation = np.array(data["rotation"], dtype=np.float64).squeeze()
        translation = np.array(data["translation"], dtype=np.float64).squeeze()
        stereo_error = data["RMSE"]

        logging.info(
            f"Loading stereo data for ports {self.portA} and {self.portB}: {data}"
        )

        return rotation, translation

    def build_projection_matrices(self):

        # camA orientation will define the global frame of reference, so
        # translation vector is [0,0,0]
        rot_trans_A = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
        mtx_A = self.camera_A.camera_matrix
        self.proj_A = mtx_A @ rot_trans_A #projection matrix for CamA

        rot_B = self.camera_B.rotation
        trans_B = np.array([[t] for t in self.camera_B.translation])
        rot_trans_B = np.concatenate([rot_B, trans_B], axis = -1)
        mtx_B = self.camera_B.camera_matrix
        self.proj_B = mtx_B @ rot_trans_B #projection matrix for C2

    def get_3D_point(self, point_A, point_B):
        
        
        u_A = common_points[f"loc_img_x_{self.portA}"]    
        v_A = common_points[f"loc_img_y_{self.portA}"]    
        u_B = common_points[f"loc_img_x_{self.portB}"]    
        v_B = common_points[f"loc_img_y_{self.portB}"]    

        # point_A = np.array([u_A, v_A]).squeeze()
        # point_B = np.array([u_B, v_B]).squeeze()

        A = [point_A[1]*self.proj_A[2,:] - self.proj_A[1,:],
             self.proj_A[0,:] - point_A[0]*self.proj_A[2,:],
             point_B[1]*self.proj_B[2,:] - self.proj_B[1,:],
             self.proj_B[0,:] - point_B[0]*self.proj_B[2,:]
            ]
        A = np.array(A).reshape((4,4))
 
        B = A.transpose() @ A
        U, s, Vh = scipy.linalg.svd(B, full_matrices = False)
        # logging.debug("You are doing it, man. You are doing it.")
        # logging.debug(common_points)
        coord_3D = Vh[3,0:3]/Vh[3,3] 
        return coord_3D


if __name__ == "__main__":

    from src.recording.recorded_stream import RecordedStreamPool
    from src.cameras.synchronizer import Synchronizer
    from src.calibration.charuco import Charuco
    from src.triangulate.common_point_finder import CommonPointFinder
    from src.calibration.corner_tracker import CornerTracker

    # set the location for the sample data used for testing
    repo = Path(__file__).parent.parent.parent
    print(repo)
    video_directory = Path(
        repo, "src", "triangulate", "sample_data", "stereo_track_charuco"
    )

    # create playback streams to provide to synchronizer
    ports = [0, 1]
    recorded_stream_pool = RecordedStreamPool(ports, video_directory)
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
    recorded_stream_pool.play_videos()

    # create a corner tracker to locate board corners
    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide=0.0525, inverted=True
    )
    trackr = CornerTracker(charuco)

    # create a commmon point finder to grab charuco corners shared between the pair of ports
    pairs = [(0, 1)]
    locatr = CommonPointFinder(
        synchronizer=syncr,
        pairs=pairs,
        tracker=trackr,
    )

    sample_config_path = str(Path(video_directory.parent, "config.toml"))
    triangulatr = StereoTriangulator(0, 1, sample_config_path)

    while True:
        common_points = locatr.paired_points_q.get()
        for index, row in common_points.iterrows():
            point_A = (row["loc_img_x_0"], row["loc_img_y_0"])
            point_B = (row["loc_img_x_1"], row["loc_img_y_1"])

            print(triangulatr.get_3D_point(point_A, point_B))
