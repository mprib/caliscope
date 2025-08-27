# %%
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
from numba.typed import Dict as NumbaDict
from numpy.typing import NDArray

import caliscope.logger

logger = caliscope.logger.get(__name__)
CAMERA_PARAM_COUNT = 6


@dataclass
class CameraData:
    """
    A place to hold the calibration data associated with a camera that has been populated from a config file.
    For use with final setting of the array and triangulation, but no actual camera management.
    Not loving the way this is implemented as an adjunct to the Camera object, but here we are
    """

    port: int
    size: list[int]
    rotation_count: int = 0
    error: float | None = None  # the RMSE of reprojection associated with the intrinsic calibration
    matrix: np.ndarray | None = None
    distortions: np.ndarray | None = None  #
    exposure: int | None = None
    grid_count: int | None = None
    ignore: bool = False
    verified_resolutions: np.ndarray | None = None
    translation: np.ndarray | None = None  # camera relative to world
    rotation: np.ndarray | None = None  # camera relative to world

    @property
    def transformation(self):
        """
        Rotation and translation combined
        """
        assert self.rotation is not None and self.translation is not None

        t = np.hstack([self.rotation, np.expand_dims(self.translation, 1)])
        t = np.vstack([t, np.array([0, 0, 0, 1], np.float32)])
        return t

    @transformation.setter
    def transformation(self, t: np.ndarray):
        self.rotation = t[0:3, 0:3]
        self.translation = t[0:3, 3]
        logger.info(f"Rotation and Translation being updated to {self.rotation} and {self.translation}")

    @property
    def projection_matrix(self):
        assert self.matrix is not None and self.transformation is not None
        return self.matrix @ self.transformation[0:3, :]

    def extrinsics_to_vector(self):
        """
        Converts camera parameters to a numpy vector for use with bundle adjustment.
        """
        # rotation of the camera relative to the world
        assert self.rotation is not None and self.translation is not None
        rotation_rodrigues = cv2.Rodrigues(self.rotation)[0]  # elements 0,1,2
        port_param = np.hstack([rotation_rodrigues[:, 0], self.translation])

        return port_param

    def extrinsics_from_vector(self, row):
        """
        Takes a vector in the same format that is output of `extrinsics_to_vector` and updates the camera
        """

        # convert back to world frame of reference
        self.rotation = cv2.Rodrigues(row[0:3])[0]
        self.translation = np.array([row[3:6]], dtype=np.float64)[0]

    def get_display_data(self) -> OrderedDict:
        # Extracting camera matrix parameters
        if self.matrix is not None:
            fx, fy = self.matrix[0, 0], self.matrix[1, 1]
            cx, cy = self.matrix[0, 2], self.matrix[1, 2]
        else:
            fx, fy = None, None
            cx, cy = None, None

        # Extracting distortion coefficients
        if self.distortions is not None:
            k1, k2, p1, p2, k3 = self.distortions.ravel()[:5]
        else:
            k1, k2, p1, p2, k3 = None, None, None, None, None

        def round_or_none(value, places):
            if value is None:
                return None
            else:
                return round(value, places)

        # Creating the dictionary with OrderedDict
        camera_display_dict = OrderedDict(
            [
                ("size", self.size),
                ("RMSE", self.error),
                ("Grid_Count", self.grid_count),
                ("rotation_count", self.rotation_count),
                (
                    "intrinsic_parameters",
                    OrderedDict(
                        [
                            ("focal_length_x", round_or_none(fx, 2)),
                            ("focal_length_y", round_or_none(fy, 2)),
                            ("optical_center_x", round_or_none(cx, 2)),
                            ("optical_center_y", round_or_none(cy, 2)),
                        ]
                    ),
                ),
                (
                    "distortion_coefficients",
                    OrderedDict(
                        [
                            ("radial_k1", round_or_none(k1, 2)),
                            ("radial_k2", round_or_none(k2, 2)),
                            ("radial_k3", round_or_none(k3, 2)),
                            ("tangential_p1", round_or_none(p1, 2)),
                            ("tangential_p2", round_or_none(p2, 2)),
                        ]
                    ),
                ),
            ]
        )

        return camera_display_dict

    def erase_calibration_data(self):
        self.error = None
        self.matrix = None
        self.distortions = None
        self.grid_count = None
        self.translation = None
        self.rotation = None


@dataclass
class CameraArray:
    """
    A data structure to hold a dictionary of all CameraData objects,
    providing views for accessing all, posed, or unposed cameras.
    """

    cameras: Dict[int, CameraData]

    @property
    def posed_cameras(self) -> Dict[int, CameraData]:
        """Returns a view of cameras that have extrinsic data (pose)."""
        return {
            port: cam for port, cam in self.cameras.items() if cam.rotation is not None and cam.translation is not None
        }

    @property
    def unposed_cameras(self) -> Dict[int, CameraData]:
        """Returns a view of cameras that are missing extrinsic data (pose)."""
        return {port: cam for port, cam in self.cameras.items() if cam.rotation is None or cam.translation is None}

    @property
    def posed_port_to_index(self) -> Dict[int, int]:
        """
        Maps the port to an index for *posed and non-ignored* cameras.
        This is used for ordering parameters for optimization routines.
        The value is re-calculated on each access to ensure it is always fresh.
        """
        # CRITICAL: This operates on `posed_cameras` to get the set of cameras
        # eligible for optimization.
        eligible_ports = [port for port, cam in self.posed_cameras.items() if not cam.ignore]
        eligible_ports.sort()  # Important for deterministic behavior
        return {port: i for i, port in enumerate(eligible_ports)}

    @property
    def posed_index_to_port(self) -> Dict[int, int]:
        """
        Maps an index back to a port for *posed and non-ignored* cameras.
        The value is re-calculated on each access to ensure it is always fresh.
        """
        return {value: key for key, value in self.posed_port_to_index.items()}

    def get_extrinsic_params(self) -> NDArray | None:
        """
        Builds the extrinsic parameter vector for all *posed* cameras.
        Returns None if no cameras are posed and not ignored.
        """
        # The index_port property already filters for posed and non-ignored cameras
        ordered_ports = self.posed_index_to_port.keys()

        if not ordered_ports:
            return None

        # Build the params in the order defined by index_port
        params_list = []
        for index in sorted(ordered_ports):
            port = self.posed_index_to_port[index]
            cam = self.cameras[port]
            params_list.append(cam.extrinsics_to_vector())

        return np.vstack(params_list)

    def update_extrinsic_params(self, least_sq_result_x: NDArray) -> None:
        """Updates extrinsic parameters from an optimization result vector."""
        indices_to_update = self.posed_index_to_port
        n_cameras = len(indices_to_update)

        if n_cameras == 0:
            logger.warning("Tried to update extrinsics, but no posed cameras were found to update.")
            return

        n_cam_param = 6  # 6 DoF
        flat_camera_params = least_sq_result_x[0 : n_cameras * n_cam_param]
        new_camera_params = flat_camera_params.reshape(n_cameras, n_cam_param)

        for index, cam_vec in enumerate(new_camera_params):
            port = indices_to_update[index]
            # When updating, we modify the original camera object in self.cameras
            self.cameras[port].extrinsics_from_vector(cam_vec)

    # Note: I've updated the docstrings on these to be more precise
    def all_extrinsics_calibrated(self) -> bool:
        """Checks if ALL cameras in the array have a pose."""
        if not self.cameras:
            return True
        return not self.unposed_cameras

    def all_intrinsics_calibrated(self) -> bool:
        """Checks if ALL cameras in the array have intrinsic data."""
        return all(cam.matrix is not None and cam.distortions is not None for cam in self.cameras.values())

    @property
    def projection_matrices(self):
        """Generates projection matrices for *posed and non-ignored* cameras only."""
        logger.info("Creating projection matrices for posed and non-ignored cameras.")
        # Note: This NumbaDict should only contain cameras used in optimization
        proj_mat = NumbaDict()  # type: ignore
        for port in self.posed_port_to_index.keys():  # port_index keys are posed and not ignored
            proj_mat[port] = self.cameras[port].projection_matrix

        return proj_mat
