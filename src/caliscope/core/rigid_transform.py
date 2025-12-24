# src/caliscope/core/rigid_transform.py
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import orthogonal_procrustes
from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import WorldPoints


def estimate_rigid_transform(
    source_points: NDArray[np.float64], target_points: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Estimate the optimal rigid transformation (rotation + translation) that aligns
    source_points to target_points using the Kabsch algorithm.

    Args:
        source_points: Nx3 array of points in source coordinate frame
        target_points: Nx3 array of points in target coordinate frame

    Returns:
        4x4 homogeneous transformation matrix
    """
    if source_points.shape != target_points.shape:
        raise ValueError(f"Point arrays must have same shape, got {source_points.shape} and {target_points.shape}")

    if source_points.shape[0] < 3:
        raise ValueError(f"Need at least 3 points for rigid transform, got {source_points.shape[0]}")

    # Center the points
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute optimal rotation using SVD
    R, _ = orthogonal_procrustes(source_centered, target_centered)

    # Compute translation
    t = target_centroid - R @ source_centroid

    # Build homogeneous transformation matrix
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R
    transform[:3, 3] = t

    return transform


def apply_rigid_transform(
    camera_array: CameraArray, world_points: WorldPoints, transform_matrix: NDArray[np.float64]
) -> tuple[CameraArray, WorldPoints]:
    """
    Apply a rigid transformation to both camera extrinsics and world points.

    Args:
        camera_array: CameraArray to transform (poses updated in-place)
        world_points: WorldPoints to transform
        transform_matrix: 4x4 homogeneous transformation matrix

    Returns:
        Tuple of (updated_camera_array, new_world_points)
    """
    if transform_matrix.shape != (4, 4):
        raise ValueError(f"Transform matrix must be 4x4, got {transform_matrix.shape}")

    # Transform world points
    points_3d = world_points.points
    homogeneous_points = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    transformed_points = (transform_matrix @ homogeneous_points.T).T[:, :3]

    new_world_df = world_points.df.copy()
    new_world_df[["x_coord", "y_coord", "z_coord"]] = transformed_points
    new_world_points = WorldPoints(new_world_df)

    # Transform camera extrinsics
    # Camera pose is T_cam_world (transform from world to camera frame)
    # We want to update it to T_cam_world' = T_cam_world * T_world_world'
    # where T_world_world' is the inverse of the object-to-world transform
    world_to_object_transform = np.linalg.inv(transform_matrix)

    for port, camera_data in camera_array.posed_cameras.items():
        # Current camera transformation (world to camera)
        current_transform = camera_data.transformation

        # New transformation: apply world shift then camera transform
        new_transform = current_transform @ world_to_object_transform

        camera_data.transformation = new_transform

    return camera_array, new_world_points
