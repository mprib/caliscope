#%%

import calicam.logger

logger = calicam.logger.get(__name__)

from pathlib import Path
import pickle
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.optimize import least_squares
import pandas as pd

from calicam.calibration.capture_volume.point_estimates import PointEstimates
from calicam.cameras.camera_array import CameraArray

CAMERA_PARAM_COUNT = 6


@dataclass
class CaptureVolume:
    camera_array: CameraArray
    point_history: PointEstimates

    def save(self, output_path):
        with open(Path(output_path), "wb") as file:
            pickle.dump(self, file)

    def get_vectorized_params(self):
        """
        Convert the parameters of the camera array and the point estimates into one long array.
        This is the required data format of the least squares optimization
        """
        camera_params = self.camera_array.get_extrinsic_params()
        combined = np.hstack((camera_params.ravel(), self.point_history.obj.ravel()))

        return combined

    def get_xy_reprojection_error(self):
        vectorized_params = self.get_vectorized_params()
        error = xy_reprojection_error(vectorized_params, self)

        return error

    # Mac: Start here tomorrow. This code was copied over but not revised to account for its new position.
    # This is a substantial refactor of high level objects that will substantially simplify their interaction
    # but it's going to be an adventure getting this to run again
    def optimize(self, output_path=None):
        # Original example taken from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

        initial_param_estimate = self.get_vectorized_params()

        # get a snapshot of where things are at the start
        initial_xy_error = xy_reprojection_error(initial_param_estimate, self)

        print(
            f"Prior to bundle adjustment, RMSE is: {rms_reproj_error(initial_xy_error)}"
        )

        # save out this snapshot if path provided
        if output_path is not None:
            self.save(Path(output_path, "pre_optimized_capture_volume.pkl"))

        self.least_sq_result = least_squares(
            xy_reprojection_error,
            initial_param_estimate,
            jac_sparsity=self.point_history.get_sparsity_pattern(),
            verbose=2,
            x_scale="jac",
            loss="linear",
            ftol=1e-8,
            method="trf",
            args=(
                self,
            ),  # xy_reprojection error takes the vectorized param estimates as first arg and capture volume as second
        )

        self.camera_array.update_extrinsic_params(self.least_sq_result.x)
        self.point_history.update_obj_xyz(self.least_sq_result.x)

        if output_path is not None:
            self.save(Path(output_path, "post_optimized_capture_volume.pkl"))

        print(
            f"Following bundle adjustment, RMSE is: {rms_reproj_error(self.least_sq_result.fun)}"
        )

    # def get_summary_df(self, label: str):

    #     array_data_xy_error = self.xy_reprojection_error.reshape(-1, 2)
    #     # build out error as singular distance

    #     xyz = self.get_xyz_points()

    #     euclidean_distance_error = np.sqrt(np.sum(array_data_xy_error**2, axis=1))
    #     row_count = euclidean_distance_error.shape[0]

    #     array_data_dict = {
    #         "label": [label] * row_count,
    #         "camera": self.point_history.camera_indices.tolist(),
    #         "sync_index": self.point_history.sync_indices.astype(int).tolist(),
    #         "charuco_id": self.point_history.point_id.tolist(),
    #         "img_x": self.point_history.img[:, 0].tolist(),
    #         "img_y": self.point_history.img[:, 1].tolist(),
    #         "reproj_error_x": array_data_xy_error[:, 0].tolist(),
    #         "reproj_error_y": array_data_xy_error[:, 1].tolist(),
    #         "reproj_error": euclidean_distance_error.tolist(),
    #         "obj_id": self.point_history.obj_indices.tolist(),
    #         "obj_x": xyz[self.point_history.obj_indices][:, 0].tolist(),
    #         "obj_y": xyz[self.point_history.obj_indices][:, 1].tolist(),
    #         "obj_z": xyz[self.point_history.obj_indices][:, 2].tolist(),
    #     }

    #     summarized_data = pd.DataFrame(array_data_dict).astype(
    #         {"sync_index": "int32", "charuco_id": "int32", "obj_id": "int32"}
    #     )
    #     return summarized_data


def xy_reprojection_error(current_param_estimates, capture_volume: CaptureVolume):
    """
    current_param_estimates: the current iteration of the vector that was originally initialized for the x0 input of least squares

    This function exists outside of the CaptureVolume class because the first argument must be the vector of parameters
    that is being adjusted by the least_squares optimization.

    """

    # Create one combined array primarily to make sure all calculations line up
    ## unpack the working estimates of the camera parameters (could be extr. or intr.)
    camera_params = current_param_estimates[
        : capture_volume.point_history.n_cameras * CAMERA_PARAM_COUNT
    ].reshape((capture_volume.point_history.n_cameras, CAMERA_PARAM_COUNT))

    ## similarly unpack the 3d point location estimates
    points_3d = current_param_estimates[
        capture_volume.point_history.n_cameras * CAMERA_PARAM_COUNT :
    ].reshape((capture_volume.point_history.n_obj_points, 3))

    ## create zero columns as placeholders for the reprojected 2d points
    rows = capture_volume.point_history.camera_indices.shape[0]
    blanks = np.zeros((rows, 2), dtype=np.float64)

    ## hstack all these arrays for ease of reference
    points_3d_and_2d = np.hstack(
        [
            np.array([capture_volume.point_history.camera_indices]).T,
            points_3d[capture_volume.point_history.obj_indices],
            capture_volume.point_history.img,
            blanks,
        ]
    )

    # iterate across cameras...while this injects a loop in the residual function
    # it should scale linearly with the number of cameras...a tradeoff for stable
    # and explicit calculations...
    for port, cam in capture_volume.camera_array.cameras.items():
        cam_points = np.where(capture_volume.point_history.camera_indices == port)
        object_points = points_3d_and_2d[cam_points][:, 1:4]

        cam_matrix = cam.camera_matrix
        rvec = camera_params[port][0:3]
        tvec = camera_params[port][3:6]
        distortion = cam.distortion[0]  # this may need some cleanup...

        # get the projection of the 2d points on the image plane; ignore the jacobian
        cam_proj_points, _jac = cv2.projectPoints(
            object_points.astype(np.float64), rvec, tvec, cam_matrix, distortion
        )

        points_3d_and_2d[cam_points, 6:8] = cam_proj_points[:, 0, :]

    points_proj = points_3d_and_2d[:, 6:8]

    # reshape the x,y reprojection error to a single vector
    return (points_proj - capture_volume.point_history.img).ravel()


def rms_reproj_error(xy_reproj_error):

    xy_reproj_error = xy_reproj_error.reshape(-1, 2)
    euclidean_distance_error = np.sqrt(np.sum(xy_reproj_error**2, axis=1))
    rmse = np.sqrt(np.mean(euclidean_distance_error**2))
    logger.info(f"Optimization run with {xy_reproj_error.shape[0]} image points")
    logger.info(f"RMSE of reprojection is {rmse}")
    return rmse


if __name__ == "__main__":
    # if True:
    from calicam import __root__
    from calicam.cameras.camera_array_builder import CameraArrayBuilder
    from calicam.calibration.capture_volume.point_estimates import (
        get_point_history,
    )

    session_directory = Path(__root__, "tests", "5_cameras")
    point_data_csv_path = Path(session_directory, "recording", "point_data.csv")

    config_path = Path(session_directory, "config.toml")
    array_builder = CameraArrayBuilder(config_path)
    camera_array = array_builder.get_camera_array()
    point_history = get_point_history(camera_array, point_data_csv_path)

    print(f"Optimizing initial camera array configuration ")

    capture_volume = CaptureVolume(camera_array, point_history)
    capture_volume.optimize(output_path=Path(session_directory, "recording"))

# %%
