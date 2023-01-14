import logging

LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_FILE = r"log\bundle_adjust_functions.log"
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)
from pathlib import Path

import sys
import cv2
import numpy as np
import pandas as pd

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import time

from src.cameras.camera_array import CameraArray, CameraArrayBuilder

# note I'm changing this from 9 as I'm reducing the values being solved for
CAMERA_PARAM_COUNT = 6


def get_camera_params(camera_array):
    """for each camera build the CAMERA_PARAM_COUNT element parameter index
    camera_params with shape (n_cameras, CAMERA_PARAM_COUNT)
    contains initial estimates of parameters for all cameras.
    First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula),
    next 3 components form a translation vector
    """
    camera_params = None
    for port, cam in camera_array.cameras.items():
        port_param = cam.to_vector()
        if camera_params is None:
            camera_params = port_param
        else:
            camera_params = np.vstack([camera_params, port_param])

    return camera_params


def get_points_2d_df(points_csv_path):
    points_df = pd.read_csv(points_csv_path)

    points_2d_port_A = points_df[
        ["port_A", "bundle", "id", "x_A_raw", "y_A_raw"]
    ].rename(
        columns={
            "port_A": "camera",
            "bundle": "bundle_index",
            "id": "point_id",
            "x_A_raw": "x_2d",
            "y_A_raw": "y_2d",
        }
    )

    points_2d_port_B = points_df[
        ["port_B", "bundle", "id", "x_B_raw", "y_B_raw"]
    ].rename(
        columns={
            "port_B": "camera",
            "bundle": "bundle_index",
            "id": "point_id",
            "x_B_raw": "x_2d",
            "y_B_raw": "y_2d",
        }
    )

    points_2d_df = (
        pd.concat([points_2d_port_A, points_2d_port_B])
        .drop_duplicates()
        .sort_values(["bundle_index", "point_id", "camera"])
        .rename(columns={"bundle_index": "bundle"})
    )
    return points_2d_df


# get 3d points with indices to merge back into points_2d
def get_points_3d_df(points_csv_path):
    points_df = pd.read_csv(points_csv_path)
    points_3d_df = (
        points_df[["bundle", "id", "pair", "x_pos", "y_pos", "z_pos"]]
        .sort_values(["bundle", "id"])
        .groupby(["bundle", "id"])
        .agg({"x_pos": "mean", "y_pos": "mean", "z_pos": "mean", "pair": "size"})
        .rename(
            columns={"pair": "count", "x_pos": "x_3d", "y_pos": "y_3d", "z_pos": "z_3d"}
        )
        .reset_index()
        .reset_index()
        .rename(columns={"index": "index_3d", "id": "point_id"})
    )
    return points_3d_df


def get_bundle_adjust_params(points_2d_df: pd.DataFrame, points_3d_df: pd.DataFrame):

    merged_point_data = (
        points_2d_df.merge(points_3d_df, how="left", on=["bundle", "point_id"])
        .sort_values(["camera", "bundle", "point_id"])
        .dropna()
    )

    camera_indices = np.array(merged_point_data["camera"], dtype=np.int64)
    point_indices = np.array(merged_point_data["index_3d"], dtype=np.int64)
    points_2d = np.array(merged_point_data[["x_2d", "y_2d"]])
    points_3d = np.array(points_3d_df[["x_3d", "y_3d", "z_3d"]])
    n_points = points_3d.shape[0]

    return camera_indices, point_indices, points_2d, points_3d, n_points


def reprojection_error(
    params, n_cameras, n_points, camera_indices, point_indices, points_2d, camera_array
):

    # reshape the camera parameter estimates into one row per camera
    camera_params = params[: n_cameras * CAMERA_PARAM_COUNT].reshape(
        (n_cameras, CAMERA_PARAM_COUNT)
    )
    
    # reshape the 3d points from vector to nx3
    points_3d = params[n_cameras * CAMERA_PARAM_COUNT :].reshape((n_points, 3))

    rows = camera_indices.shape[0]
    blanks = np.zeros((rows, 2), dtype=np.float64)
    points_3d_and_2d = np.hstack(
        [np.array([camera_indices]).T, points_3d[point_indices], points_2d, blanks]
    )

    for port, cam in camera_array.cameras.items():
        cam_points = np.where(camera_indices == port)
        object_points = points_3d_and_2d[cam_points][:, 1:4]
        rvec = camera_params[port][0:3]
        tvec = camera_params[port][3:6]
        cam_matrix = cam.camera_matrix
        distortion = cam.distortion[0]  # this may need some cleanup...

        # get the projection of the 2d points on the image plane and ignore the jacobian
        cam_proj_points, _jac = cv2.projectPoints(
            object_points.astype(np.float64), rvec, tvec, cam_matrix, distortion
        )

        points_3d_and_2d[cam_points, 6:8] = cam_proj_points[:, 0, :]

    points_proj = points_3d_and_2d[:, 6:8]
    # points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def get_sparsity_pattern(n_cameras, n_points, camera_indices, point_indices):
    """provide the sparsity structure for the Jacobian (elements that are not zero)"""
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


def bundle_adjust(camera_array: CameraArray, points_csv_path: Path):
# Original example taken from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

    camera_params = get_camera_params(camera_array)

    points_2d_df = get_points_2d_df(points_csv_path)
    points_3d_df = get_points_3d_df(points_csv_path)

    (
        camera_indices,
        point_indices,
        points_2d,
        points_3d,
        n_points,
    ) = get_bundle_adjust_params(points_2d_df, points_3d_df)

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
    objective_value = reprojection_error(
        initial_estimate,
        n_cameras,
        n_points,
        camera_indices,
        point_indices,
        points_2d,
        camera_array,
    )

    sparsity_pattern = get_sparsity_pattern(
        n_cameras, n_points, camera_indices, point_indices
    )

    t0 = time.time()
    logging.info(f"Start time of bundle adjustment calculations is {t0}")
    optimized = least_squares(
        reprojection_error,
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
