import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from caliscope.core.capture_volume.point_estimates import PointEstimates
from caliscope.core.capture_volume.set_origin_functions import (
    get_board_origin_transform,
)
from caliscope.core.charuco import Charuco
from caliscope.cameras.camera_array import CameraArray

logger = logging.getLogger(__name__)

CAMERA_PARAM_COUNT = 6  # rotation and translation parameters
OPTIMIZATION_LOOPS = 0


@dataclass
class CaptureVolume:
    camera_array: CameraArray
    _point_estimates: PointEstimates  # Internal storage for point data
    stage: int = 0
    _normalized_points: np.ndarray | None = None
    origin_sync_index: int | None = None

    def __post__init__(self):
        logger.info("Creating capture volume from estimated camera array and stereotriangulated points...")

    def _save(self, directory: Path, descriptor: str | None = None):
        """
        Convenience function to facilitate visual checks through the stages of processing using the GUI
        """
        if descriptor is None:
            pkl_name = "capture_volume_stage_" + str(self.stage) + ".pkl"
        else:
            pkl_name = "capture_volume_stage_" + str(self.stage) + "_" + descriptor + ".pkl"
        logger.info(f"Saving stage {str(self.stage)} capture volume to {directory}")
        with open(Path(directory, pkl_name), "wb") as file:
            pickle.dump(self, file)

    def get_vectorized_params(self):
        """
        Convert the parameters of the camera array and the point estimates into one long array.
        This is the required data format of the least squares optimization
        """
        camera_params = self.camera_array.get_extrinsic_params()
        if camera_params is None:
            raise ValueError("No posed cameras available for optimization")
        combined = np.hstack((camera_params.ravel(), self._point_estimates.obj.ravel()))

        return combined

    @property
    def point_estimates(self) -> PointEstimates:
        """Public accessor for the point estimates data."""
        return self._point_estimates

    @point_estimates.setter
    def point_estimates(self, value: PointEstimates):
        """
        Setter for point estimates that automatically invalidates dependent caches.
        This ensures that if points are filtered, the normalized points and
        previous optimization results are cleared.
        """
        logger.info("PointEstimates have been updated. Invalidating dependent caches.")
        self._point_estimates = value
        self._normalized_points = None
        if hasattr(self, "least_sq_result"):
            delattr(self, "least_sq_result")

    @property
    def normalized_points(self) -> np.ndarray:
        """
        Lazily computes and caches the normalized (undistorted) image points.

        This is a key optimization. The expensive undistortion for all points
        is performed only once and the result is cached. The bundle adjustment's
        hot loop can then work with this pre-computed data.
        """
        if self._normalized_points is None:
            logger.info("Computing and caching normalized points for optimization...")
            self._normalized_points = np.zeros_like(self._point_estimates.img)

            # Iterate through each camera that has a pose and will be used in the optimization
            for port, cam in self.camera_array.posed_cameras.items():
                # Get the zero-based index for this camera
                camera_index = self.camera_array.posed_port_to_index.get(port)
                if camera_index is None:
                    continue  # Skip if camera is not part of the optimization set

                # Find all observations that belong to the current camera
                obs_indices_for_cam = np.where(self._point_estimates.camera_indices == camera_index)[0]
                if obs_indices_for_cam.size > 0:
                    img_points_for_cam = self._point_estimates.img[obs_indices_for_cam]
                    undistorted = cam.undistort_points(img_points_for_cam, output="normalized")
                    self._normalized_points[obs_indices_for_cam] = undistorted
        return self._normalized_points

    @property
    def rmse(self):
        # This map is needed to translate optimization indices back to port numbers
        index_to_port = self.camera_array.posed_index_to_port
        # Always calculate RMSE in pixels for an intuitive and consistent metric.
        xy_pixel_error = self.get_xy_reprojection_error()
        rmse = rms_reproj_error(xy_pixel_error, self._point_estimates.camera_indices, index_to_port)

        return rmse

    def get_rmse_summary(self):
        rmse_string = f"RMSE of Reprojection Overall: {round(self.rmse['overall'], 2)}\n"
        rmse_string += "    by camera:\n"
        for key, value in self.rmse.items():
            if key == "overall":
                pass
            else:
                rmse_string += f"    {key: >9}: {round(float(value), 2)}\n"

        return rmse_string

    def get_xy_reprojection_error(self):
        """Calculates reprojection error against original pixel coordinates."""
        vectorized_params = self.get_vectorized_params()
        # Pass `use_normalized=False` to get error in pixels
        error = xy_reprojection_error(vectorized_params, self, use_normalized=False)

        return error

    def optimize(self, ftol=1e-8):
        # Original example taken from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

        initial_param_estimate = self.get_vectorized_params()

        logger.info(f"Beginning bundle adjustment to calculated stage {self.stage + 1}")
        # Ensure normalized points are computed before optimization
        _ = self.normalized_points

        self.least_sq_result = least_squares(
            xy_reprojection_error,  # This will now use the optimized version
            initial_param_estimate,
            jac_sparsity=self._point_estimates.get_sparsity_pattern(),
            verbose=2,
            x_scale="jac",
            loss="linear",
            ftol=ftol,
            method="trf",
            # By default, xy_reprojection_error will use normalized points
            args=(self,),
        )

        self.camera_array.update_extrinsic_params(self.least_sq_result.x)
        self._point_estimates.update_obj_xyz(self.least_sq_result.x)
        self.stage += 1

        logger.info(f"Following bundle adjustment (stage {str(self.stage)}), pixel RMSE is: {self.rmse['overall']:.4f}")

    def get_xyz_points(self):
        """Get 3d positions arrived at by bundle adjustment"""
        n_posed_cameras = len(self.camera_array.posed_port_to_index)

        camera_params_size = n_posed_cameras * CAMERA_PARAM_COUNT
        xyz = self.get_vectorized_params()[camera_params_size:]
        xyz = xyz.reshape(-1, 3)

        return xyz

    def apply_transform(self, transform_matrix: np.ndarray):
        # update 3d point estimates
        xyz = self._point_estimates.obj
        scale = np.expand_dims(np.ones(xyz.shape[0]), 1)
        xyzh = np.hstack([xyz, scale])

        new_origin_xyzh = np.matmul(np.linalg.inv(transform_matrix), xyzh.T).T
        self._point_estimates.obj = new_origin_xyzh[:, 0:3]

        # update camera array
        for port, camera_data in self.camera_array.posed_cameras.items():
            camera_data.transformation = np.matmul(camera_data.transformation, transform_matrix)

    def set_origin_to_board(self, sync_index, charuco: Charuco):
        """
        Find the pose of the charuco (rvec and tvec) from a given frame
        Transform stereopairs and 3d point estimates for this new origin
        """
        self.origin_sync_index = sync_index

        logger.info(f"Capture volume origin set to board position at sync index {sync_index}")

        origin_transform = get_board_origin_transform(self.camera_array, self._point_estimates, sync_index, charuco)
        self.apply_transform(origin_transform)

    def rotate(self, direction: str):
        """Applies a 90-degree rotation around the specified axis."""
        transformations = {
            "x+": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float),
            "x-": np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float),
            "y+": np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float),
            "y-": np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=float),
            "z+": np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
            "z-": np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
        }
        self.apply_transform(transformations[direction])


def xy_reprojection_error(current_param_estimates, capture_volume: CaptureVolume, use_normalized: bool = True):
    """
    Calculates the reprojection error for the bundle adjustment optimization.

    This function compares the original 2D point detections with the 2D reprojections
    of the estimated 3D points using the current camera parameter estimates.

    Args:
        current_param_estimates: A 1D numpy array containing the flattened camera
                                 parameters and 3D point coordinates being optimized.
        capture_volume: The CaptureVolume object containing the static camera
                        and point data.
        use_normalized: If True, compares against pre-computed normalized points
                        and uses a perfect camera model (fast). If False, compares
                        against original image points and uses the full camera model (slow).

    Returns:
        A flattened 1D numpy array of residuals (the difference between observed
        and reprojected 2D points).
    """

    global OPTIMIZATION_LOOPS

    # --- Select the target 2D points and camera model based on the mode ---
    if use_normalized:
        target_image_points = capture_volume.normalized_points
    else:
        target_image_points = capture_volume._point_estimates.img

    # 1. Unpack the optimization vector into meaningful variables
    n_cams = capture_volume._point_estimates.n_cameras
    n_pts = capture_volume._point_estimates.n_obj_points

    # Unpack the camera extrinsics (R|T) being optimized
    camera_params = current_param_estimates[: n_cams * CAMERA_PARAM_COUNT].reshape((n_cams, CAMERA_PARAM_COUNT))

    # Unpack the 3D point coordinates being optimized
    points_3d = current_param_estimates[n_cams * CAMERA_PARAM_COUNT :].reshape((n_pts, 3))

    # Select the 3D points corresponding to each 2D observation
    object_points_to_project = points_3d[capture_volume._point_estimates.obj_indices]

    # Initialize an array to store the new reprojected points
    points_proj = np.zeros_like(target_image_points)

    # 2. Iterate through only the POSED cameras to calculate reprojection
    for port, cam in capture_volume.camera_array.posed_cameras.items():
        # Get the correct zero-based index for this camera's parameters
        camera_index = capture_volume.camera_array.posed_port_to_index[port]

        # Find all observations that belong to the current camera
        # This is the key fix: comparing index to index
        obs_indices_for_cam = np.where(capture_volume._point_estimates.camera_indices == camera_index)[0]

        # --- ROBUSTNESS GUARD ---
        # If this camera has no points to project, skip it
        if obs_indices_for_cam.size == 0:
            continue

        # Get the 3D points corresponding to this camera's observations
        object_points = object_points_to_project[obs_indices_for_cam]

        # Get the camera parameters for projection
        rvec = camera_params[camera_index, 0:3]
        tvec = camera_params[camera_index, 3:6]

        # --- Select camera intrinsics based on the mode ---
        if use_normalized:
            # OPTIMIZED PATH: Use the 'perfect' camera model. Computational benefits.
            cam_matrix = np.identity(3)
            dist_coeffs = np.zeros(5)  # No distortion in normalized space
        else:
            # SLOW PATH: Use the camera's true intrinsics. Needed for final pixel error.
            if cam.matrix is None or cam.distortions is None:
                raise ValueError(f"Camera {port} missing intrinsics for pixel-mode reprojection")
            cam_matrix = cam.matrix
            dist_coeffs = cam.distortions

        reprojected_pts, _ = cv2.projectPoints(object_points, rvec, tvec, cam_matrix, dist_coeffs)

        # Store the results in our output array
        points_proj[obs_indices_for_cam] = reprojected_pts.reshape(-1, 2)

    OPTIMIZATION_LOOPS += 1

    # 3. Calculate the final error
    # The error is the difference between the original 2D detections and our new reprojections.
    residual = (points_proj - target_image_points).ravel()

    logger.debug(f"OPTIMIZATION LOOPS: {OPTIMIZATION_LOOPS}")

    return residual


def rms_reproj_error(
    xy_reproj_error: np.ndarray, camera_indices: np.ndarray, index_to_port: dict[int, int]
) -> dict[str, float]:
    """
    Calculates the root-mean-square reprojection error, overall and per camera.

    Args:
        xy_reproj_error: A numpy array of shape (N, 2) or (N*2,) containing
                         the x and y reprojection errors for N points.
        camera_indices: A numpy array of shape (N,) mapping each error to a
                        zero-based camera index.
        index_to_port: A dictionary mapping the zero-based camera index to the
                       human-readable camera port number.

    Returns:
        A dictionary with the "overall" RMSE and an entry for each camera port.
    """
    rmse = {}
    xy_reproj_error = xy_reproj_error.reshape(-1, 2)
    euclidean_distance_error = np.sqrt(np.sum(xy_reproj_error**2, axis=1))
    rmse["overall"] = float(np.sqrt(np.mean(euclidean_distance_error**2)))

    # Iterate through unique camera indices present in the data
    for index in np.unique(camera_indices):
        camera_errors = euclidean_distance_error[camera_indices == index]
        # Use the map to get the human-readable port number for the dictionary key
        port = index_to_port[index]
        rmse[str(port)] = float(np.sqrt(np.mean(camera_errors**2)))

    return rmse
