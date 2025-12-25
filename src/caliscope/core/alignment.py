from dataclasses import dataclass
import logging
import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.point_data import WorldPoints

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimilarityTransform:
    """
    Immutable representation of a similarity transform: target = s * (R @ source) + t

    Attributes:
        rotation: 3x3 proper rotation matrix (det = +1)
        translation: 3-vector in target coordinate units
        scale: uniform scale factor (target_units / source_units)
    """

    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    scale: float

    def __post_init__(self):
        """Validate transform components after creation."""
        # Validate rotation shape and properties
        if self.rotation.shape != (3, 3):
            raise ValueError(f"Rotation must be 3x3, got {self.rotation.shape}")

        det = np.linalg.det(self.rotation)
        if not np.isclose(det, 1.0, atol=1e-6):
            raise ValueError(f"Rotation must be proper (det=+1), got det={det:.6f}")

        if not np.allclose(self.rotation @ self.rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("Rotation matrix must be orthogonal")

        # Validate translation
        if self.translation.shape != (3,):
            raise ValueError(f"Translation must be 3-vector, got {self.translation.shape}")

        # Validate scale
        if self.scale <= 0:
            raise ValueError(f"Scale must be positive, got {self.scale}")

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply transform to Nx3 points: target = s * (R @ points) + t

        Args:
            points: Nx3 array of source points

        Returns:
            Nx3 array of transformed points
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must be Nx3 array, got shape {points.shape}")

        return self.scale * (self.rotation @ points.T).T + self.translation

    @property
    def inverse(self) -> "SimilarityTransform":
        """Return the inverse transform."""
        inv_rotation = self.rotation.T  # Inverse of rotation is its transpose
        inv_scale = 1.0 / self.scale
        inv_translation = -inv_scale * (inv_rotation @ self.translation)

        return SimilarityTransform(inv_rotation, inv_translation, inv_scale)

    @property
    def matrix(self) -> NDArray[np.float64]:
        """
        4x4 homogeneous transformation matrix [[s*R, t], [0, 1]].
        Useful for composition with other transformations.
        """
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix


def estimate_similarity_transform(
    source_points: NDArray[np.float64],
    target_points: NDArray[np.float64],
) -> SimilarityTransform:
    """
    Estimate optimal similarity transform using Umeyama's algorithm.

    Finds s, R, t that minimize ||target - (s * (R @ source) + t)||²

    Args:
        source_points: Nx3 array of points in source coordinate frame
        target_points: Nx3 array of points in target coordinate frame

    Returns:
        SimilarityTransform containing rotation, translation, and scale

    Raises:
        ValueError: If input arrays are invalid or too few points provided
    """
    # Input validation
    if source_points.shape != target_points.shape:
        raise ValueError(f"Point arrays must have same shape, got {source_points.shape} and {target_points.shape}")

    if source_points.shape[0] < 3:
        raise ValueError(f"Need at least 3 points for similarity transform, got {source_points.shape[0]}")

    if source_points.shape[1] != 3:
        raise ValueError(f"Points must be 3D (Nx3), got shape {source_points.shape}")

    if np.any(np.isnan(source_points)) or np.any(np.isnan(target_points)):
        raise ValueError("Input points cannot contain NaN values")

    # Center the points
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute covariance matrix
    H = source_centered.T @ target_centered

    # SVD for optimal rotation
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (determinant +1, not reflection)
    if np.linalg.det(Vt.T @ U.T) < 0:
        Vt[-1, :] *= -1

    rotation = Vt.T @ U.T

    # Compute scale
    source_variance = np.sum(source_centered**2)
    scale = np.sum(target_centered * (rotation @ source_centered.T).T) / source_variance

    # Compute translation
    translation = target_centroid - scale * (rotation @ source_centroid)

    # Create and validate transform
    try:
        return SimilarityTransform(rotation, translation, float(scale))
    except ValueError as e:
        raise RuntimeError(f"Estimated transform is invalid: {e}")


def apply_similarity_transform(
    camera_array: CameraArray,
    world_points: WorldPoints,
    transform: SimilarityTransform,
) -> tuple[CameraArray, WorldPoints]:
    """
    Apply similarity transform to camera array and world points.

    Transforms world points from source to target coordinate frame, and updates
    camera extrinsics accordingly. Returns new instances (immutable).

    Args:
        camera_array: CameraArray to transform
        world_points: WorldPoints to transform
        transform: SimilarityTransform to apply

    Returns:
        Tuple of (new_camera_array, new_world_points)
    """
    # Transform world points
    points_3d = world_points.points
    transformed_points = transform.apply(points_3d)

    # Create new WorldPoints instance
    world_df = world_points.df.copy()
    world_df[["x_coord", "y_coord", "z_coord"]] = transformed_points
    new_world_points = WorldPoints(world_df)

    # Transform camera extrinsics
    # Camera pose is T_cam_world (world → camera)
    # We want: T_cam_world_new = T_cam_world_old @ T_world_old_world_new
    # where T_world_old_world_new is the inverse of the similarity transform
    world_to_world_transform = transform.inverse.matrix

    # Build new camera dictionary
    new_cameras = {}
    for port, camera_data in camera_array.cameras.items():
        # Copy camera data (shallow copy is fine since we're replacing transformation)
        new_camera_data = CameraData(
            port=camera_data.port,
            size=camera_data.size,
            rotation_count=camera_data.rotation_count,
            error=camera_data.error,
            matrix=camera_data.matrix,
            distortions=camera_data.distortions,
            exposure=camera_data.exposure,
            grid_count=camera_data.grid_count,
            ignore=camera_data.ignore,
            fisheye=camera_data.fisheye,
        )

        # If camera has extrinsics, transform them
        if camera_data.rotation is not None and camera_data.translation is not None:
            current_transform = camera_data.transformation
            new_transform = current_transform @ world_to_world_transform

            # Update the new camera data
            new_camera_data.transformation = new_transform

        new_cameras[port] = new_camera_data

    new_camera_array = CameraArray(cameras=new_cameras)

    return new_camera_array, new_world_points
