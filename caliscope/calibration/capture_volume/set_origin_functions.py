# %%


import logging

import cv2
import numpy as np
import scipy

from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.charuco import Charuco
from caliscope.cameras.camera_array import CameraArray, CameraData

logger = logging.getLogger(__name__)


# Proceeding with basic idea that these functions will go into CaptureVolume.
# so use capture_volume here which may just become self later on.


def get_world_corners_xyz(point_estimates: PointEstimates, sync_index: int) -> np.ndarray:
    """
    returns the estimated x,y,z position of the board corners at the given sync index
    note that the array is ordered according to the charuco id
    """
    sync_indices = point_estimates.sync_indices  # convienent shortening

    charuco_ids = point_estimates.point_id[sync_indices == sync_index]
    unique_charuco_id = np.unique(charuco_ids)
    unique_charuco_id.sort()

    # pull out the 3d point estimate indexes associated with the chosen sync_index
    # note that this will include duplicates
    obj_indices = point_estimates.obj_indices[sync_indices == sync_index]
    # now get the actual x,y,z estimate associated with these unique charucos
    obj_xyz = point_estimates.obj[obj_indices]

    sorter = np.argsort(charuco_ids)
    # need to get charuco ids associated with the 3 point positions
    unique_charuco_xyz_index = sorter[np.searchsorted(charuco_ids, unique_charuco_id, sorter=sorter)]
    world_corners_xyz = obj_xyz[unique_charuco_xyz_index]
    return world_corners_xyz


def get_board_corners_xyz(point_estimates: PointEstimates, sync_index: int, charuco: Charuco) -> np.ndarray:
    """
    Returns corner positions in board world (x,y,0) for the corners with estimated point
    coordinates at the give sync_index
    note that the array is ordered according to the charuco id
    """
    sync_indices = point_estimates.sync_indices  # convienent shortening
    charuco_ids = point_estimates.point_id[sync_indices == sync_index]
    unique_charuco_id = np.unique(charuco_ids)
    unique_charuco_id.sort()

    board_corners_xyz = charuco.board.getChessboardCorners()[unique_charuco_id]
    return board_corners_xyz


def get_anchor_cameras(camera_array: CameraArray, point_estimates: PointEstimates, sync_index: int) -> list:
    """
    Returns the camera data objects that have an actual view of the board
    at the sync index and therefore can be used to estimate the pose from pnp

    Weird results will happen if the camera has its back turned to the board and
    the corners are projected into it.

    """
    sync_indices = point_estimates.sync_indices  # convienent shortening
    indices_of_cameras_w_view = point_estimates.camera_indices[sync_indices == sync_index]
    camera_indices, camera_counts = np.unique(indices_of_cameras_w_view, return_counts=True)

    logger.info(f"Indices of cameras with view of calibration board are {camera_indices}")
    logger.info(f"CameraArray.cameras.keys() = {camera_array.cameras.keys()}")
    logger.info(f"CameraArray.posed_index_to_port = {camera_array.posed_index_to_port}")

    anchor_cameras = []
    for index in camera_indices:
        logger.info(f"index of port is {index}")
        port = camera_array.posed_index_to_port[index]
        logger.info(f"port is {port}")
        cam: CameraData = camera_array.cameras[port]
        anchor_cameras.append(cam)

    return anchor_cameras


def get_rvec_tvec_from_board_pose(
    camera_array: CameraArray,
    point_estimates: PointEstimates,
    sync_index: int,
    charuco: Charuco,
):
    world_corners_xyz = get_world_corners_xyz(point_estimates, sync_index)
    board_corners_xyz = get_board_corners_xyz(point_estimates, sync_index, charuco)
    anchor_cameras = get_anchor_cameras(camera_array, point_estimates, sync_index)

    rvecs = []
    tvecs = []

    for camera in anchor_cameras:
        charuco_image_points, jacobian = cv2.projectPoints(
            world_corners_xyz,
            rvec=camera.rotation,
            tvec=camera.translation,
            cameraMatrix=camera.matrix,
            distCoeffs=np.array([0, 0, 0, 0, 0], dtype=np.float32),  # because points are via bundle adj., no distortion
        )

        # use solvepnp to estimate the pose of the camera relative to the board
        # this provides a good estimate of rotation, but not of translation
        retval, rvec, tvec = cv2.solvePnP(
            board_corners_xyz,
            charuco_image_points,
            cameraMatrix=camera.matrix,
            distCoeffs=np.array([0, 0, 0, 0, 0], dtype=np.float32),
        )

        anchor_board_transform = rvec_tvec_to_transform(rvec, tvec)

        # back into the shift in the world change of origin implied by the
        # pose of the camera relative to the board given its previous
        # pose in the old frame of reference
        origin_shift_transform = np.matmul(np.linalg.inv(camera.transformation), anchor_board_transform)

        rvec, tvec = transform_to_rvec_tvec(origin_shift_transform)

        rvecs.append(rvec)
        tvecs.append(tvec)

    mean_rvec = mean_vec(rvecs)
    mean_tvec = mean_vec(tvecs)

    # get mean values
    return mean_rvec, mean_tvec


def mean_vec(vecs):
    hstacked_vec = None

    for vec in vecs:
        if hstacked_vec is None:
            hstacked_vec = vec
        else:
            hstacked_vec = np.hstack([hstacked_vec, vec])

    mean_vec = np.mean(hstacked_vec, axis=1)
    mean_vec = np.expand_dims(mean_vec, axis=1)

    return mean_vec


def transform_to_rvec_tvec(transformation: np.ndarray):
    rot_matrix = transformation[0:3, 0:3]
    rvec = cv2.Rodrigues(rot_matrix)[0]
    tvec = np.expand_dims(transformation[0:3, 3], axis=1)
    return rvec, tvec


def rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    # might send a rotation matrix into here so check
    # that rvec is a rodrigues vector before converting
    if len(rvec.shape) == 1 or rvec.shape[1] == 1:
        rvec = cv2.Rodrigues(rvec)[0]

    # if tvec doesn't have the extra dimension, add it
    if len(tvec.shape) == 1:
        tvec = np.expand_dims(tvec, axis=1)

    transform = np.hstack([rvec, tvec])
    transform = np.vstack([transform, np.array([0, 0, 0, 1], np.float32)])
    return transform


def world_board_distance(tvec_xyz: np.ndarray, good_rvec: np.ndarray, raw_world_xyz, board_corners_xyz):
    scale = np.expand_dims(np.ones(raw_world_xyz.shape[0]), 1)
    raw_world_xyzh = np.hstack([raw_world_xyz, scale])

    origin_shift_transform = rvec_tvec_to_transform(good_rvec, tvec_xyz)

    new_origin_xyzh = np.matmul(np.linalg.inv(origin_shift_transform), raw_world_xyzh.T).T
    new_origin_xyz = new_origin_xyzh[:, 0:3]

    delta_xyz = new_origin_xyz - board_corners_xyz
    logger.info(f"Delta_xyz is {delta_xyz}")
    return delta_xyz.ravel()


def get_board_origin_transform(
    camera_array: CameraArray,
    point_estimates: PointEstimates,
    sync_index: int,
    charuco: Charuco,
):
    """
    Returns the 4x4 transformation matrix that will shift the capture volumes camera and
    point data to be represented with an origin that aligns with the origin of the charuco board.

    This is the primary function that is exposed from this file to the CaptureVolume class
    """
    # get initial approximation of the transformation to apply
    good_rvec, poor_tvec = get_rvec_tvec_from_board_pose(
        camera_array=camera_array,
        point_estimates=point_estimates,
        sync_index=sync_index,
        charuco=charuco,
    )

    world_board = get_world_corners_xyz(point_estimates, sync_index)
    target_board = get_board_corners_xyz(point_estimates, sync_index, charuco)
    initial_tvec = poor_tvec[:, 0]  # prep for least_squares array

    least_sq_result = scipy.optimize.least_squares(
        world_board_distance, initial_tvec, args=(good_rvec, world_board, target_board)
    )

    optimal_tvec = least_sq_result.x
    final_transform = rvec_tvec_to_transform(good_rvec, optimal_tvec)

    return final_transform
