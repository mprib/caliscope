import logging

LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_FILE = r"log\bundle_adjust_functions.log"
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from pathlib import Path
import cv2
import numpy as np

from dataclasses import dataclass
from scipy.optimize import least_squares
import time

from calicam.cameras.camera_array import CameraArray
from calicam.calibration.bundle_adjustment.point_data import PointData


CAMERA_PARAM_COUNT = 6


def xy_reprojection_error(
    current_param_estimates,
    camera_array,
    point_data
):
    """
    current_param_estimates: the current iteration of the vector that was originally initialized for the x0 input of least squares
    """

    # Create one combined array primarily to make sure all calculations line up
    ## unpack the working estimates of the camera 6dof
    camera_params = current_param_estimates[: point_data.n_cameras * CAMERA_PARAM_COUNT].reshape(
        (point_data.n_cameras, CAMERA_PARAM_COUNT)
    )

    ## similarly unpack the 3d points estimates
    points_3d = current_param_estimates[point_data.n_cameras * CAMERA_PARAM_COUNT :].reshape(
        (point_data.n_obj_points, 3)
    )

    ## created zero columns as placeholders for the reprojected 2d points
    rows = point_data.camera_indices.shape[0]
    blanks = np.zeros((rows, 2), dtype=np.float64)

    ## hstack all these arrays for ease of reference
    points_3d_and_2d = np.hstack(
        [np.array([point_data.camera_indices]).T, points_3d[point_data.obj_indices], point_data.img, blanks]
    )

    # iterate across cameras...while this injects a loop in the residual function
    # it should scale linearly with the number of cameras...a tradeoff for stable
    # and explicit calculations...
    for port, cam in camera_array.cameras.items():
        cam_points = np.where(point_data.camera_indices == port)
        object_points = points_3d_and_2d[cam_points][:, 1:4]
        rvec = camera_params[port][0:3]
        tvec = camera_params[port][3:6]
        cam_matrix = cam.camera_matrix
        distortion = cam.distortion[0]  # this may need some cleanup...

        # get the projection of the 2d points on the image plane; ignore the jacobian
        cam_proj_points, _jac = cv2.projectPoints(
            object_points.astype(np.float64), rvec, tvec, cam_matrix, distortion
        )

        points_3d_and_2d[cam_points, 6:8] = cam_proj_points[:, 0, :]

    points_proj = points_3d_and_2d[:, 6:8]

    # reshape the x,y reprojection error to a single vector
    return (points_proj - point_data.img).ravel()



def bundle_adjust(camera_array: CameraArray, point_data: PointData):
    # Original example taken from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    camera_params = camera_array.get_extrinsic_params()
    # n_cameras = camera_params.shape[0]

    # n_obj_points = point_data.obj.shape[0]

    n = CAMERA_PARAM_COUNT * point_data.n_cameras + 3 * point_data.n_obj_points
    m = 2 * point_data.n_img_points

    logging.info(f"n_cameras: {point_data.n_cameras}")
    logging.info(f"n_points: {point_data.n_obj_points}")
    logging.info(f"Total number of parameters: {n}")
    logging.info(f"Total number of residuals: {m}")

    initial_param_estimate = np.hstack((camera_params.ravel(), point_data.obj.ravel()))

    # sparsity_pattern = get_sparsity_pattern(point_data)

    t0 = time.time()
    logging.info(f"Start time of bundle adjustment calculations is {t0}")

    optimized = least_squares(
        xy_reprojection_error,
        initial_param_estimate,
        jac_sparsity=point_data.get_sparsity_pattern(),
        verbose=2,
        x_scale="jac",
        loss="linear",
        ftol=1e-8,
        method="trf",
        args=(
            camera_array,
            point_data,
        ),
    )

    t1 = time.time()
    logging.info(f"Completion time of bundle adjustment calculations is {t1}")
    logging.info(f"Total time to perform bundle adjustment: {t1-t0}")

    return optimized
