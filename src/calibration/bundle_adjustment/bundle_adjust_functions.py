import logging

LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_FILE = r"log\bundle_adjust_functions.log"
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from pathlib import Path
import cv2
import numpy as np

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import time

from src.cameras.camera_array import CameraArray, CameraArrayBuilder
from src.calibration.bundle_adjustment.get_init_params import (
    get_2d_3d_points,
    get_camera_params,
)


# note I'm changing this from 9 as I'm reducing the values being solved for
CAMERA_PARAM_COUNT = 6


def xy_reprojection_error(
    params, n_cameras, n_points, camera_indices, point_indices, points_2d, camera_array
):
    """
    params: the current iteration of the vector that was originally initialized for the x0 input of least squares

    """

    # Create one combined array primarily to make sure all calculations line up

    ## convert current vectorized camera parameter estimates to matrix for ease of reference
    camera_params = params[: n_cameras * CAMERA_PARAM_COUNT].reshape(
        (n_cameras, CAMERA_PARAM_COUNT)
    )

    ## similarly convert the current vectorized 3d point estimates to matrix for ease of reference
    points_3d = params[n_cameras * CAMERA_PARAM_COUNT :].reshape((n_points, 3))

    ## created zero columns as placeholders for the reprojected 2d points
    rows = camera_indices.shape[0]
    blanks = np.zeros((rows, 2), dtype=np.float64)

    ## hstack these arrays for ease of reference
    points_3d_and_2d = np.hstack(
        [np.array([camera_indices]).T, points_3d[point_indices], points_2d, blanks]
    )

    # iterate across cameras...while this injects a loop in the residual function
    # it should scale linearly with the number of cameras...a tradeoff for stable
    # and explicit calculations...
    for port, cam in camera_array.cameras.items():
        cam_points = np.where(camera_indices == port)
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
    # points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def get_sparsity_pattern(n_cameras, n_points, camera_indices, point_indices):
    """provide the sparsity structure for the Jacobian (elements that are not zero)
    n_points: number of unique 3d points; these will each have at least one but potentially more associated 2d points
    point_indices: a vector that maps the 2d points to their associated 3d point
    """
    m = camera_indices.size * 2
    n = n_cameras * CAMERA_PARAM_COUNT + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(CAMERA_PARAM_COUNT):
        A[2 * i, camera_indices * CAMERA_PARAM_COUNT + s] = 1
        A[2 * i + 1, camera_indices * CAMERA_PARAM_COUNT + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * CAMERA_PARAM_COUNT + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * CAMERA_PARAM_COUNT + point_indices * 3 + s] = 1

    return A


# TODO: #60 don't pass a csv_path...you can't really reuse this function if that is the case


def bundle_adjust(
    camera_array: CameraArray, camera_indices, point_indices, points_2d, points_3d
):
    # Original example taken from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

    # MAC: start here tomorrow. You need to figure out how to cull the points
    # based on reproj error and rerun bundle adjustment

    camera_params = get_camera_params(camera_array)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = CAMERA_PARAM_COUNT * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    logging.info(f"n_cameras: {n_cameras}")
    logging.info(f"n_points: {n_points}")
    logging.info(f"Total number of parameters: {n}")
    logging.info(f"Total number of residuals: {m}")

    initial_estimate = np.hstack((camera_params.ravel(), points_3d.ravel()))

    # test the reprojection_error...here ahead of least squares for debugging purposes
    # objective_value = xy_reprojection_error(
    #     initial_estimate,
    #     n_cameras,
    #     n_points,
    #     camera_indices,
    #     point_indices,
    #     points_2d,
    #     camera_array,
    # )

    sparsity_pattern = get_sparsity_pattern(
        n_cameras, n_points, camera_indices, point_indices
    )

    t0 = time.time()
    logging.info(f"Start time of bundle adjustment calculations is {t0}")
    optimized = least_squares(
        xy_reprojection_error,
        initial_estimate,
        jac_sparsity=sparsity_pattern,
        verbose=2,
        x_scale="jac",
        loss="linear",
        ftol=1e-8,
        method="trf",
        args=(
            n_cameras,
            n_points,
            camera_indices,
            point_indices,
            points_2d,
            camera_array,
        ),
    )
    t1 = time.time()
    logging.info(f"Completion time of bundle adjustment calculations is {t1}")
    logging.info(f"Total time to perform bundle adjustment: {t1-t0}")

    return optimized
