
import calicam.logger
logger = calicam.logger.get(__name__)

from pathlib import Path
import pickle
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.optimize import least_squares

from calicam.calibration.capture_volume.point_estimate_data import PointEstimateData
from calicam.cameras.camera_array import CameraArray

CAMERA_PARAM_COUNT = 6
@dataclass
class CaptureVolume:
    point_estimate_data: PointEstimateData
    model_params: np.ndarray  # the first argument of the residual function
    xy_reprojection_error: np.ndarray
    camera_array: CameraArray

    def save(self, output_path):
        with open(Path(output_path), "wb") as file:
            pickle.dump(self, file)

    
    def get_xyz_points(self):
        """Get 3d positions arrived at by bundle adjustment"""
        n_cameras = len(self.camera_array.cameras)
        xyz = self.model_params[n_cameras * CAMERA_PARAM_COUNT :]
        xyz = xyz.reshape(-1, 3)

        return xyz
    
    def get_summary_df(self, label:str):
        
        array_data_xy_error = self.xy_reprojection_error.reshape(-1, 2)
        # build out error as singular distance

        xyz = self.get_xyz_points()

        euclidean_distance_error = np.sqrt(np.sum(array_data_xy_error**2, axis=1))
        row_count = euclidean_distance_error.shape[0]

        array_data_dict = {
            "label": [label] * row_count,
            "camera": self.point_estimate_data.camera_indices_full.tolist(),
            "sync_index": self.point_estimate_data.sync_indices.astype(int).tolist(),
            "charuco_id": self.point_estimate_data.corner_id.tolist(),
            "img_x": self.point_estimate_data.img_full[:, 0].tolist(),
            "img_y": self.point_estimate_data.img_full[:, 1].tolist(),
            "reproj_error_x": array_data_xy_error[:, 0].tolist(),
            "reproj_error_y": array_data_xy_error[:, 1].tolist(),
            "reproj_error": euclidean_distance_error.tolist(),
            "obj_id": self.point_estimate_data.obj_indices.tolist(),
            "obj_x": xyz[self.point_estimate_data.obj_indices_full][:, 0].tolist(),
            "obj_y": xyz[self.point_estimate_data.obj_indices_full][:, 1].tolist(),
            "obj_z": xyz[self.point_estimate_data.obj_indices_full][:, 2].tolist(),
        }

        summarized_data = (pd.DataFrame(array_data_dict)
                            .astype({"sync_index":'int32', "charuco_id":"int32", "obj_id":"int32"})
        )
        return summarized_data
    
    
    
    def bundle_adjust(self, point_estimate_data: PointEstimateData, output_path=None):
        # Original example taken from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

        camera_params = self.get_extrinsic_params()
        initial_param_estimate = np.hstack(
            (camera_params.ravel(), point_estimate_data.obj.ravel())
        )

        # get a snapshot of where things are at the start
        initial_xy_error = xy_reprojection_error(
            initial_param_estimate,
            self,
            point_estimate_data,
        )

        print(
            f"Prior to bundle adjustment, RMSE is: {rms_reproj_error(initial_xy_error)}"
        )

        # save out this snapshot if path provided
        if output_path is not None:
            diagnostic_data = CaptureVolume(
                point_estimate_data, initial_param_estimate, initial_xy_error, self
            )
            diagnostic_data.save(Path(output_path, "before_bund_adj.pkl"))

        self.least_sq_result = least_squares(
            xy_reprojection_error,
            initial_param_estimate,
            jac_sparsity=point_estimate_data.get_sparsity_pattern(),
            verbose=2,
            x_scale="jac",
            loss="linear",
            ftol=1e-8,
            method="trf",
            args=(
                self,
                point_estimate_data,
            ),
        )

        if output_path is not None:
            diagnostic_data = CaptureVolume(
                point_estimate_data, self.least_sq_result.x, self.least_sq_result.fun, self
            )
            diagnostic_data.save(Path(output_path, "after_bund_adj.pkl"))

        print(
            f"Following bundle adjustment, RMSE is: {rms_reproj_error(self.least_sq_result.fun)}"
        )
        return self.least_sq_result
    
    
    
    
def xy_reprojection_error(
    current_param_estimates,
    camera_array: CameraArray,
    bund_adj_data: PointEstimateData,
):
    """
    current_param_estimates: the current iteration of the vector that was originally initialized for the x0 input of least squares
    """

    # Create one combined array primarily to make sure all calculations line up
    ## unpack the working estimates of the camera parameters (could be extr. or intr.)
    camera_params = current_param_estimates[
        : bund_adj_data.n_cameras * CAMERA_PARAM_COUNT
    ].reshape((bund_adj_data.n_cameras, CAMERA_PARAM_COUNT))

    ## similarly unpack the 3d point location estimates
    points_3d = current_param_estimates[
        bund_adj_data.n_cameras * CAMERA_PARAM_COUNT :
    ].reshape((bund_adj_data.n_obj_points, 3))

    ## create zero columns as placeholders for the reprojected 2d points
    rows = bund_adj_data.camera_indices.shape[0]
    blanks = np.zeros((rows, 2), dtype=np.float64)

    ## hstack all these arrays for ease of reference
    points_3d_and_2d = np.hstack(
        [
            np.array([bund_adj_data.camera_indices]).T,
            points_3d[bund_adj_data.obj_indices],
            bund_adj_data.img,
            blanks,
        ]
    )

    # iterate across cameras...while this injects a loop in the residual function
    # it should scale linearly with the number of cameras...a tradeoff for stable
    # and explicit calculations...
    for port, cam in camera_array.cameras.items():
        cam_points = np.where(bund_adj_data.camera_indices == port)
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
    return (points_proj - bund_adj_data.img).ravel()



def rms_reproj_error(xy_reproj_error):

    xy_reproj_error = xy_reproj_error.reshape(-1, 2)
    euclidean_distance_error = np.sqrt(np.sum(xy_reproj_error**2, axis=1))
    rmse = np.sqrt(np.mean(euclidean_distance_error**2))
    logger.info(f"Optimization run with {xy_reproj_error.shape[0]} image points")
    logger.info(f"RMSE of reprojection is {rmse}")
    return rmse