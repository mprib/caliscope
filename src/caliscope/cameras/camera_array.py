# %%
from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import cv2
import numpy as np
import rtoml
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
CAMERA_PARAM_COUNT = 6


@dataclass
class CameraData:
    """Single camera with calibration parameters.

    Calibration-relevant fields:
        cam_id, size, matrix, distortions, rotation, translation, fisheye

    Workspace fields (ignore in scripting context):
        rotation_count, exposure, grid_count, ignore
    """

    cam_id: int
    size: tuple[int, int]
    rotation_count: int = 0
    error: float | None = None  # the RMSE of reprojection associated with the intrinsic calibration
    matrix: np.ndarray | None = None
    distortions: np.ndarray | None = None  #
    exposure: int | None = None
    grid_count: int | None = None
    ignore: bool = False
    translation: np.ndarray | None = None  # camera relative to world
    rotation: np.ndarray | None = None  # camera relative to world
    fisheye: bool = False  # default to standard camera model

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
    def normalized_projection_matrix(self):
        assert self.matrix is not None and self.transformation is not None
        return self.transformation[0:3, :]

    def extrinsics_to_vector(self):
        """
        Converts camera parameters to a numpy vector for use with bundle adjustment.
        """
        # rotation of the camera relative to the world
        assert self.rotation is not None and self.translation is not None
        rotation_rodrigues = cv2.Rodrigues(self.rotation)[0]  # elements 0,1,2
        cam_param = np.hstack([rotation_rodrigues[:, 0], self.translation])

        return cam_param

    def extrinsics_from_vector(self, row):
        """
        Takes a vector in the same format that is output of `extrinsics_to_vector` and updates the camera
        """

        # convert back to world frame of reference
        self.rotation = cv2.Rodrigues(row[0:3])[0]
        self.translation = np.array([row[3:6]], dtype=np.float64)[0]

    def undistort_points(self, points: NDArray, *, output: Literal["normalized", "pixels"]) -> NDArray:
        """
        Remove lens distortion from 2D image points.

        Using normalized coordinates for triangulation and bundle adjustment
        improves numerical conditioning of the Jacobian/Hessian matrices.
        See Triggs et al., "Bundle Adjustment - A Modern Synthesis" (2000).

        Args:
            points: (N, 2) array of distorted points in pixel coordinates
            output: Coordinate system for result
                - 'normalized': Identity K matrix, for triangulation/bundle adjustment
                - 'pixels': Camera's K matrix, for reprojection error in pixel units

        Returns:
            (N, 2) array of undistorted points in the specified coordinate system
        """
        if self.matrix is None or self.distortions is None:
            raise ValueError(f"Camera {self.cam_id} lacks intrinsic calibration; cannot undistort points.")

        # OpenCV functions require points in shape (N, 1, 2) and float32 type
        points_reshaped = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 1, 2)

        # Select output projection matrix based on requested coordinate system
        if output == "normalized":
            # Identity matrix maps to normalized image plane (principal point at origin, f=1)
            projection_matrix = np.identity(3)
        else:  # output == "pixels"
            projection_matrix = self.matrix

        if self.fisheye:
            undistorted_points = cv2.fisheye.undistortPoints(
                points_reshaped, self.matrix, self.distortions, P=projection_matrix
            )
        else:
            undistorted_points = cv2.undistortPoints(
                points_reshaped, self.matrix, self.distortions, P=projection_matrix
            )

        return undistorted_points.reshape(-1, 2)

    def undistort_frame(self, frame: NDArray) -> NDArray:
        """Undistort a frame using original matrix geometry with cached remap tables.

        Uses self.matrix as the output projection matrix, guaranteeing that
        points from undistort_points(output='pixels') align with this frame.

        Output has same dimensions as input. Depending on distortion magnitude:
        - Barrel distortion (k1 < 0): content may extend beyond edges (clipped)
        - Pincushion distortion (k1 > 0): black borders may appear at edges

        Remap tables are cached on first call and reused for efficiency (~10x faster
        than cv2.undistort per frame). Cache is invalidated if frame size changes.

        For display visualization, use LensModelVisualizer instead.
        """
        if self.matrix is None or self.distortions is None:
            raise ValueError(f"Camera {self.cam_id} lacks intrinsic calibration; cannot undistort frame.")

        h, w = frame.shape[:2]
        frame_size = (w, h)

        # Check if we need to (re)compute remap tables
        if not hasattr(self, "_remap_cache") or self._remap_cache.get("size") != frame_size:
            if self.fisheye:
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    self.matrix, self.distortions, np.eye(3), self.matrix, frame_size, cv2.CV_16SC2
                )
            else:
                map1, map2 = cv2.initUndistortRectifyMap(
                    self.matrix, self.distortions, np.eye(3), self.matrix, frame_size, cv2.CV_16SC2
                )
            self._remap_cache = {"size": frame_size, "map1": map1, "map2": map2}

        return cv2.remap(frame, self._remap_cache["map1"], self._remap_cache["map2"], cv2.INTER_LINEAR)

    def get_display_data(self) -> OrderedDict:
        # Extracting camera matrix parameters
        logger.info("Extracting camera parameters... ")
        logger.info(f"Matrix: {self.matrix}")
        logger.info(f"Distortion = {self.distortions}")

        if self.matrix is not None:
            fx, fy = self.matrix[0, 0], self.matrix[1, 1]
            cx, cy = self.matrix[0, 2], self.matrix[1, 2]
        else:
            fx, fy = None, None
            cx, cy = None, None

        def round_or_none(value, places):
            if value is None:
                return None
            else:
                return round(value, places)

        # Conditionally create the distortion dictionary based on camera model
        distortion_coeffs_dict = OrderedDict()
        if self.distortions is not None:
            coeffs = self.distortions.ravel().tolist()
            logger.info(f"Unpacking distortion coefficients: {coeffs}")
            if self.fisheye:
                # Fisheye model uses 4 coefficients: k1, k2, k3, k4
                k1, k2, k3, k4 = coeffs
                distortion_coeffs_dict["radial_k1"] = round_or_none(k1, 2)
                distortion_coeffs_dict["radial_k2"] = round_or_none(k2, 2)
                distortion_coeffs_dict["radial_k3"] = round_or_none(k3, 2)
                distortion_coeffs_dict["radial_k4"] = round_or_none(k4, 2)
            else:
                # Standard model uses 5 coefficients: k1, k2, p1, p2, k3
                k1, k2, p1, p2, k3 = coeffs
                distortion_coeffs_dict["radial_k1"] = round_or_none(k1, 2)
                distortion_coeffs_dict["radial_k2"] = round_or_none(k2, 2)
                distortion_coeffs_dict["radial_k3"] = round_or_none(k3, 2)
                distortion_coeffs_dict["tangential_p1"] = round_or_none(p1, 2)
                distortion_coeffs_dict["tangential_p2"] = round_or_none(p2, 2)
        else:
            # If distortions are None, populate the dictionary with Nones
            # to maintain a consistent structure for the UI.
            if self.fisheye:
                distortion_coeffs_dict = OrderedDict(
                    [("radial_k1", None), ("radial_k2", None), ("radial_k3", None), ("radial_k4", None)]
                )
            else:
                distortion_coeffs_dict = OrderedDict(
                    [
                        ("radial_k1", None),
                        ("radial_k2", None),
                        ("radial_k3", None),
                        ("tangential_p1", None),
                        ("tangential_p2", None),
                    ]
                )

        # Creating the main dictionary with the correctly structured distortion info
        camera_display_dict = OrderedDict(
            [
                ("size", self.size),
                ("RMSE", self.error),
                ("Grid_Count", self.grid_count),
                ("rotation_count", self.rotation_count),
                ("fisheye", self.fisheye),
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
                ("distortion_coefficients", distortion_coeffs_dict),
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
            cam_id: cam
            for cam_id, cam in self.cameras.items()
            if cam.rotation is not None and cam.translation is not None
        }

    @property
    def unposed_cameras(self) -> Dict[int, CameraData]:
        """Returns a view of cameras that are missing extrinsic data (pose)."""
        return {cam_id: cam for cam_id, cam in self.cameras.items() if cam.rotation is None or cam.translation is None}

    @property
    def posed_cam_id_to_index(self) -> Dict[int, int]:
        """
        Maps the cam_id to an index for *posed and non-ignored* cameras.
        This is used for ordering parameters for optimization routines.
        The value is re-calculated on each access to ensure it is always fresh.
        """
        # CRITICAL: This operates on `posed_cameras` to get the set of cameras
        # eligible for optimization.
        eligible_cam_ids = [cam_id for cam_id, cam in self.posed_cameras.items() if not cam.ignore]
        eligible_cam_ids.sort()  # Important for deterministic behavior
        return {cam_id: i for i, cam_id in enumerate(eligible_cam_ids)}

    @property
    def posed_index_to_cam_id(self) -> Dict[int, int]:
        """
        Maps an index back to a cam_id for *posed and non-ignored* cameras.
        The value is re-calculated on each access to ensure it is always fresh.
        """
        return {value: key for key, value in self.posed_cam_id_to_index.items()}

    def __getitem__(self, cam_id: int) -> CameraData:
        return self.cameras[cam_id]

    def __setitem__(self, cam_id: int, camera: CameraData) -> None:
        self.cameras[cam_id] = camera

    @classmethod
    def from_video_metadata(cls, videos: Mapping[int, Path | str]) -> CameraArray:
        """Create uncalibrated CameraArray from video file metadata.

        Reads resolution from each video via PyAV. No frames are decoded.
        """
        from caliscope.recording.video_utils import read_video_properties

        cameras = {}
        for cam_id, video_path in videos.items():
            props = read_video_properties(Path(video_path))
            cameras[cam_id] = CameraData(cam_id=cam_id, size=props["size"])
        return cls(cameras)

    @classmethod
    def from_image_sizes(cls, sizes: dict[int, tuple[int, int]]) -> CameraArray:
        """Create uncalibrated CameraArray from known image sizes.

        Args:
            sizes: Mapping of cam_id to (width, height).
        """
        cameras = {}
        for cam_id, size in sizes.items():
            cameras[cam_id] = CameraData(cam_id=cam_id, size=size)
        return cls(cameras)

    def get_extrinsic_params(self) -> NDArray | None:
        """
        Builds the extrinsic parameter vector for all *posed* cameras.
        Returns None if no cameras are posed and not ignored.
        """
        # The index_cam_id property already filters for posed and non-ignored cameras
        ordered_cam_ids = self.posed_index_to_cam_id.keys()

        if not ordered_cam_ids:
            return None

        # Build the params in the order defined by index_cam_id
        params_list = []
        for index in sorted(ordered_cam_ids):
            cam_id = self.posed_index_to_cam_id[index]
            cam = self.cameras[cam_id]
            params_list.append(cam.extrinsics_to_vector())

        return np.vstack(params_list)

    def update_extrinsic_params(self, least_sq_result_x: NDArray) -> None:
        """Updates extrinsic parameters from an optimization result vector."""
        indices_to_update = self.posed_index_to_cam_id
        n_cameras = len(indices_to_update)

        if n_cameras == 0:
            logger.warning("Tried to update extrinsics, but no posed cameras were found to update.")
            return

        n_cam_param = 6  # 6 DoF
        flat_camera_params = least_sq_result_x[0 : n_cameras * n_cam_param]
        new_camera_params = flat_camera_params.reshape(n_cameras, n_cam_param)

        for index, cam_vec in enumerate(new_camera_params):
            cam_id = indices_to_update[index]
            # When updating, we modify the original camera object in self.cameras
            self.cameras[cam_id].extrinsics_from_vector(cam_vec)

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
    def normalized_projection_matrices(self) -> dict[int, np.ndarray]:
        """Generates normalized projection matrices for posed and non-ignored cameras."""
        logger.info("Creating normalized projection matrices for posed and non-ignored cameras.")
        proj_mat: dict[int, np.ndarray] = {}
        for cam_id in self.posed_cam_id_to_index.keys():
            proj_mat[cam_id] = self.cameras[cam_id].normalized_projection_matrix
        return proj_mat

    @classmethod
    def from_toml(cls, path: Path) -> "CameraArray":
        """Load CameraArray from TOML file.

        Rotation is stored as a 3x1 Rodrigues vector in TOML but may be a 3x3
        matrix in legacy data. This method handles both formats.

        Raises:
            PersistenceError: If file doesn't exist, is invalid TOML, or contains
                             malformed data
        """
        from caliscope.persistence import PersistenceError
        from caliscope.core.toml_helpers import _list_to_array, _clean_scalar

        if not path.exists():
            raise PersistenceError(f"CameraArray file not found: {path}")

        try:
            data = rtoml.load(path)
        except Exception as e:
            raise PersistenceError(f"Failed to load CameraArray from {path}: {e}") from e

        if not data or "cameras" not in data:
            return cls({})

        cameras_dict = {}
        for cam_id_str, camera_data in data["cameras"].items():
            try:
                cam_id = int(cam_id_str)

                matrix = _list_to_array(camera_data.get("matrix"))
                distortions = _list_to_array(camera_data.get("distortions"))
                translation = _list_to_array(camera_data.get("translation"))

                rotation_raw = _list_to_array(camera_data.get("rotation"))
                if rotation_raw is not None:
                    if rotation_raw.shape == (3, 3):
                        rotation = rotation_raw
                    elif rotation_raw.shape in [(3,), (3, 1)]:
                        rotation = cv2.Rodrigues(rotation_raw)[0]
                    else:
                        raise ValueError(f"Invalid rotation shape: {rotation_raw.shape}")
                else:
                    rotation = None

                camera = CameraData(
                    cam_id=cam_id,
                    size=(camera_data["size"][0], camera_data["size"][1]),
                    rotation_count=camera_data.get("rotation_count", 0),
                    error=_clean_scalar(camera_data.get("error")),
                    matrix=matrix,
                    distortions=distortions,
                    exposure=_clean_scalar(camera_data.get("exposure")),
                    grid_count=_clean_scalar(camera_data.get("grid_count")),
                    ignore=camera_data.get("ignore", False),
                    translation=translation,
                    rotation=rotation,
                    fisheye=camera_data.get("fisheye", False),
                )
                cameras_dict[cam_id] = camera

            except Exception as e:
                raise PersistenceError(f"Failed to parse camera {cam_id_str}: {e}") from e

        return cls(cameras_dict)

    def to_toml(self, path: Path) -> None:
        """Save CameraArray to TOML file.

        Converts 3x3 rotation matrices to 3x1 Rodrigues vectors for storage.

        Raises:
            PersistenceError: If serialization or write fails
        """
        from caliscope.persistence import PersistenceError, _safe_write_toml
        from caliscope.core.toml_helpers import _array_to_list

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            cameras_data = {}
            for cam_id, camera in self.cameras.items():
                # Use `is not None` (not `.any()`) -- the .any() check would drop
                # identity rotation (all-zeros in Rodrigues form).
                rotation_for_config = None
                if camera.rotation is not None:
                    rotation_for_config = cv2.Rodrigues(camera.rotation)[0][:, 0].tolist()

                camera_dict = {
                    "cam_id": camera.cam_id,
                    "size": camera.size,
                    "rotation_count": camera.rotation_count,
                    "error": camera.error,
                    "matrix": _array_to_list(camera.matrix),
                    "distortions": _array_to_list(camera.distortions),
                    "translation": _array_to_list(camera.translation),
                    "rotation": rotation_for_config,
                    "exposure": camera.exposure,
                    "grid_count": camera.grid_count,
                    "fisheye": camera.fisheye,
                }

                # In TOML, missing key = None/Null. Prevents rtoml "null" strings.
                clean_camera_dict = {k: v for k, v in camera_dict.items() if v is not None}
                cameras_data[str(cam_id)] = clean_camera_dict

            data = {"cameras": cameras_data}
            _safe_write_toml(data, path)

        except Exception as e:
            raise PersistenceError(f"Failed to save CameraArray to {path}: {e}") from e

    def to_aniposelib_toml(self, path: Path) -> None:
        """Save CameraArray in aniposelib-compatible TOML format.

        Only exports posed cameras. Uses top-level [cam_N] sections instead of
        nested structure. Rotation stored as 3x1 Rodrigues vector.

        Raises:
            PersistenceError: If serialization or write fails
        """
        from caliscope.persistence import PersistenceError, _safe_write_toml
        from caliscope.core.toml_helpers import _array_to_list

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            data: dict[str, Any] = {}
            for cam_id, camera in self.posed_cameras.items():
                # Use `is not None` (not `.any()`) -- the .any() check would drop
                # identity rotation (all-zeros in Rodrigues form).
                rotation_rodrigues = None
                if camera.rotation is not None:
                    rotation_rodrigues = cv2.Rodrigues(camera.rotation)[0][:, 0].tolist()

                distortions_flat = camera.distortions.ravel().tolist() if camera.distortions is not None else None
                translation_flat = camera.translation.ravel().tolist() if camera.translation is not None else None

                camera_dict = {
                    "name": f"cam_{cam_id}",
                    "size": [int(camera.size[0]), int(camera.size[1])],
                    "matrix": _array_to_list(camera.matrix),
                    "distortions": distortions_flat,
                    "rotation": rotation_rodrigues,
                    "translation": translation_flat,
                }
                data[f"cam_{cam_id}"] = camera_dict

            data["metadata"] = {"adjusted": False, "error": 0.0}
            _safe_write_toml(data, path)
            logger.info(f"Saved aniposelib-compatible camera array to {path}")

        except Exception as e:
            raise PersistenceError(f"Failed to save aniposelib CameraArray to {path}: {e}") from e
