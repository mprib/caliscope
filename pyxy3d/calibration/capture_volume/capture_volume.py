#%%

import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from pathlib import Path
import pickle
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.optimize import least_squares
import pandas as pd
from time import perf_counter

from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.calibration.capture_volume.set_origin_functions import get_board_origin_transform

CAMERA_PARAM_COUNT = 6


@dataclass
class CaptureVolume:
    camera_array: CameraArray
    point_estimates: PointEstimates
    stage: int = 0
    _rmse: float = None

    def save(self, directory: Path, descriptor: str = None):
        if descriptor is None:
            pkl_name = "capture_volume_stage_" + str(self.stage) + ".pkl"
        else:

            pkl_name = (
                "capture_volume_stage_" + str(self.stage) + "_" + descriptor + ".pkl"
            )
        logger.info(f"Saving stage {str(self.stage)} capture volume to {directory}")
        with open(Path(directory, pkl_name), "wb") as file:
            pickle.dump(self, file)

    def get_vectorized_params(self):
        """
        Convert the parameters of the camera array and the point estimates into one long array.
        This is the required data format of the least squares optimization
        """
        camera_params = self.camera_array.get_extrinsic_params()
        combined = np.hstack((camera_params.ravel(), self.point_estimates.obj.ravel()))

        return combined

    @property
    def rmse(self):

        if hasattr(self, "least_sq_result"):
            rmse = rms_reproj_error(self.least_sq_result.fun)
        else:
            param_estimates = self.get_vectorized_params()
            xy_repro_error = xy_reprojection_error(param_estimates, self)
            rmse = rms_reproj_error(xy_repro_error)

        return rmse

    def get_xy_reprojection_error(self):
        vectorized_params = self.get_vectorized_params()
        error = xy_reprojection_error(vectorized_params, self)

        return error

    def optimize(self):
        # Original example taken from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

        initial_param_estimate = self.get_vectorized_params()

        # get a snapshot of where things are at the start
        initial_xy_error = xy_reprojection_error(initial_param_estimate, self)

        # logger.info(
        #     f"Prior to bundle adjustment (stage {str(self.stage)}), RMSE is: {self.rmse}"
        # )
        logger.info(f"Beginning bundle adjustment to calculated stage {self.stage+1}")
        self.least_sq_result = least_squares(
            xy_reprojection_error,
            initial_param_estimate,
            jac_sparsity=self.point_estimates.get_sparsity_pattern(),
            verbose=2,
            x_scale="jac",
            loss="linear",
            ftol=1e-8,
            method="trf",
            # xy_reprojection error takes the vectorized param estimates as first arg and capture volume as second
            args=(self,),
        )

        self.camera_array.update_extrinsic_params(self.least_sq_result.x)
        self.point_estimates.update_obj_xyz(self.least_sq_result.x)
        self.stage += 1

        logger.info(
            f"Following bundle adjustment (stage {str(self.stage)}), RMSE is: {self.rmse}"
        )

    def get_xyz_points(self):
        """Get 3d positions arrived at by bundle adjustment"""
        n_cameras = len(self.camera_array.cameras)
        xyz = self.get_vectorized_params()[n_cameras * CAMERA_PARAM_COUNT :]
        xyz = xyz.reshape(-1, 3)

        return xyz

    def shift_origin(self, origin_shift_transform: np.ndarray):

        # update 3d point estimates
        xyz = self.point_estimates.obj
        scale = np.expand_dims(np.ones(xyz.shape[0]), 1)
        xyzh = np.hstack([xyz, scale])

        new_origin_xyzh = np.matmul(np.linalg.inv(origin_shift_transform), xyzh.T).T
        self.point_estimates.obj = new_origin_xyzh[:, 0:3]

        # update camera array
        for port, camera_data in self.camera_array.cameras.items():
            camera_data.transformation = np.matmul(
                camera_data.transformation, origin_shift_transform
            )


    def set_origin_to_board(self, sync_index, charuco: Charuco):
        """
        Find the pose of the charuco (rvec and tvec) from a given frame
        Transform stereopairs and 3d point estimates for this new origin
        """

        origin_transform = get_board_origin_transform(self.camera_array,
                                                      self.point_estimates,
                                                      sync_index, 
                                                      charuco)
        self.shift_origin(origin_transform)


def xy_reprojection_error(current_param_estimates, capture_volume: CaptureVolume):
    """
    current_param_estimates: the current iteration of the vector that was originally initialized for the x0 input of least squares

    This function exists outside of the CaptureVolume class because the first argument must be the vector of parameters
    that is being adjusted by the least_squares optimization.

    """
    # Create one combined array primarily to make sure all calculations line up
    ## unpack the working estimates of the camera parameters (could be extr. or intr.)
    camera_params = current_param_estimates[
        : capture_volume.point_estimates.n_cameras * CAMERA_PARAM_COUNT
    ].reshape((capture_volume.point_estimates.n_cameras, CAMERA_PARAM_COUNT))

    ## similarly unpack the 3d point location estimates
    points_3d = current_param_estimates[
        capture_volume.point_estimates.n_cameras * CAMERA_PARAM_COUNT :
    ].reshape((capture_volume.point_estimates.n_obj_points, 3))

    ## create zero columns as placeholders for the reprojected 2d points
    rows = capture_volume.point_estimates.camera_indices.shape[0]
    blanks = np.zeros((rows, 2), dtype=np.float64)

    ## hstack all these arrays for ease of reference
    points_3d_and_2d = np.hstack(
        [
            np.array([capture_volume.point_estimates.camera_indices]).T,
            points_3d[capture_volume.point_estimates.obj_indices],
            capture_volume.point_estimates.img,
            blanks,
        ]
    )

    # iterate across cameras...while this injects a loop in the residual function
    # it should scale linearly with the number of cameras...a tradeoff for stable
    # and explicit calculations...
    for port, cam in capture_volume.camera_array.cameras.items():
        cam_points = np.where(capture_volume.point_estimates.camera_indices == port)
        object_points = points_3d_and_2d[cam_points][:, 1:4]

        cam_matrix = cam.matrix
        rvec = camera_params[port][0:3]
        tvec = camera_params[port][3:6]
        distortions = cam.distortions

        # get the projection of the 2d points on the image plane; ignore the jacobian
        cam_proj_points, _jac = cv2.projectPoints(
            object_points.astype(np.float64), rvec, tvec, cam_matrix, distortions
        )

        points_3d_and_2d[cam_points, 6:8] = cam_proj_points[:, 0, :]

    points_proj = points_3d_and_2d[:, 6:8]

    xy_reprojection_error = (points_proj - capture_volume.point_estimates.img).ravel()
    if round(perf_counter(), 2) * 10 % 1 == 0:
        logger.info(
            f"Optimizing... RMSE of reprojection = {rms_reproj_error(xy_reprojection_error)}"
        )

    # reshape the x,y reprojection error to a single vector
    return xy_reprojection_error


def rms_reproj_error(xy_reproj_error):

    xy_reproj_error = xy_reproj_error.reshape(-1, 2)
    euclidean_distance_error = np.sqrt(np.sum(xy_reproj_error**2, axis=1))
    rmse = np.sqrt(np.mean(euclidean_distance_error**2))
    # logger.info(f"Optimization run with {xy_reproj_error.shape[0]} image points")
    # logger.info(f"RMSE of reprojection is {rmse}")
    return rmse


if __name__ == "__main__":
    # if True:
    from pyxy3d import __root__
    from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer
    from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
        get_point_estimates,
    )

    session_directory = Path(__root__, "tests", "4_cameras_endofday")

    point_data_csv_path = Path(session_directory, "point_data.csv")

    config_path = Path(session_directory, "config.toml")
    array_initializer = CameraArrayInitializer(config_path)
    camera_array = array_initializer.get_best_camera_array()
    point_estimates = get_point_estimates(camera_array, point_data_csv_path)

    print(f"Optimizing initial camera array configuration ")

    self = CaptureVolume(camera_array, point_estimates)
    self.save(session_directory)
    self.optimize()
    self.save(session_directory)
# %%
