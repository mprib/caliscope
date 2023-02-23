import calicam.logger
logger = calicam.logger.get(__name__)

from pathlib import Path
import numpy as np
from dataclasses import dataclass
import cv2


CAMERA_PARAM_COUNT = 6

@dataclass
class CameraData:
    """
    A place to hold the calibration data associated with a camera that has been populated from a config file.
    For use with final setting of the array and triangulation, but no actual camera management.
    """

    port: int
    resolution: tuple
    camera_matrix: np.ndarray
    error: float  # the RMSE of reprojection associated with the intrinsic calibration
    distortion: np.ndarray  #
    translation: np.ndarray
    rotation: np.ndarray

    def extrinsics_to_vector(self):
        """
        Converts camera parameters to a numpy vector for use with bundle adjustment.
        """
        # rotation of the camera relative to the world
        rotation_matrix_world = self.rotation

        # rotation of the world relative to camera
        rotation_matrix_proj = np.linalg.inv(rotation_matrix_world)
        rotation_rodrigues = cv2.Rodrigues(rotation_matrix_proj)[0]  # elements 0,1,2
        translation_world = self.translation  # elements 3,4,5
        translation_proj = translation_world * -1

        port_param = np.hstack([rotation_rodrigues[:, 0], translation_proj[:, 0]])

        return port_param

    def extrinsics_from_vector(self, row):
        """
        Takes a vector in the same format that is output of `extrinsics_to_vector` and updates the camera
        """

        # convert back to world frame of reference
        self.rotation = np.linalg.inv(cv2.Rodrigues(row[0:3])[0])
        self.translation = np.array([row[3:6] * -1], dtype=np.float64).T


@dataclass
class CameraArray:
    """The plan is that this will expand to become an interface for setting the origin.
    At the moment all it is doing is holding a dictionary of CameraData objects"""

    cameras: dict
    # least_sq_result = None # this may be a vestige of an old way of doing things. remove if commenting out doesn't break anything.
    
    def get_extrinsic_params(self):
        """for each camera build the CAMERA_PARAM_COUNT element parameter index
        camera_params with shape (n_cameras, CAMERA_PARAM_COUNT)
        contains initial estimates of parameters for all cameras.
        First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula),
        next 3 components form a translation vector
        """

        camera_params = None
        for port, cam in self.cameras.items():
            port_param = cam.extrinsics_to_vector()
            if camera_params is None:
                camera_params = port_param
            else:
                camera_params = np.vstack([camera_params, port_param])

        return camera_params

    def update_extrinsic_params(self, least_sq_result_x:np.array):

        n_cameras = len(self.cameras)
        n_cam_param = 6  # 6 DoF
        flat_camera_params = least_sq_result_x[0 : n_cameras * n_cam_param]
        new_camera_params = flat_camera_params.reshape(n_cameras, n_cam_param)

        # update camera array with new positional data
        for index in range(len(new_camera_params)):
            print(index)
            port = index  # just to be explicit
            cam_vec = new_camera_params[index, :]
            self.cameras[port].extrinsics_from_vector(cam_vec)


if __name__ == "__main__":
    from calicam.cameras.camera_array_builder import CameraArrayBuilder
    from calicam.calibration.capture_volume.point_estimates import (
        PointEstimates,
        get_point_history,
    )

    from calicam import __root__

    session_directory = Path(__root__, "tests", "5_cameras")
    config_path = Path(session_directory, "config.toml")
    array_builder = CameraArrayBuilder(config_path)
    camera_array = array_builder.get_camera_array()

