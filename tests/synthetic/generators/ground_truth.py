"""
Synthetic ground truth generation for camera calibration testing.

Provides geometrically consistent camera arrays, world points, and image projections
where all data is mathematically exact (no noise). This serves as the "answer key"
that optimization should converge toward.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.point_data import ImagePoints, WorldPoints


@dataclass(frozen=True)
class SyntheticGroundTruth:
    """
    The oracle - perfect camera poses, world points, and image observations.

    All data is geometrically consistent: projecting world_points through cameras
    produces exactly image_points (no noise). This is the "answer key" that
    optimization should converge toward.

    Attributes:
        cameras: CameraArray with perfect intrinsics and extrinsics
        world_points: WorldPoints with perfect 3D coordinates
        image_points: ImagePoints with perfect 2D projections (no noise)
    """

    cameras: CameraArray
    world_points: WorldPoints
    image_points: ImagePoints

    def with_camera_perturbation(
        self,
        rotation_sigma: float,
        translation_sigma: float,
        rng: np.random.Generator,
        fixed_ports: list[int] | None = None,
    ) -> CameraArray:
        """
        Create perturbed camera array for optimization testing.

        Perturbations are applied in the tangent space of SE(3):
        - Rotation: Gaussian noise added to Rodrigues vector, then converted back to rotation matrix
        - Translation: Gaussian noise added directly to translation vector

        Args:
            rotation_sigma: Standard deviation for rotation perturbation in radians.
                            Typical value: 0.05 rad (~2.9 degrees)
            translation_sigma: Standard deviation for translation perturbation in mm.
                               Typical value: 50.0 mm
            rng: NumPy random generator for reproducibility
            fixed_ports: Camera ports to leave unperturbed (gauge reference).
                         Default: [0] (first camera is the gauge reference)

        Returns:
            New CameraArray with perturbed extrinsics (intrinsics unchanged)
        """
        if fixed_ports is None:
            fixed_ports = [0]

        perturbed = deepcopy(self.cameras)

        for port, camera in perturbed.cameras.items():
            if port in fixed_ports:
                continue
            if camera.rotation is None or camera.translation is None:
                continue

            # Perturb rotation in Rodrigues space
            rodrigues, _ = cv2.Rodrigues(camera.rotation)
            rodrigues = rodrigues.ravel() + rng.normal(0, rotation_sigma, 3)
            camera.rotation, _ = cv2.Rodrigues(rodrigues)

            # Perturb translation
            camera.translation = camera.translation + rng.normal(0, translation_sigma, 3)

        return perturbed

    def with_image_noise(
        self,
        pixel_sigma: float,
        rng: np.random.Generator,
    ) -> ImagePoints:
        """
        Add Gaussian noise to image point observations.

        Args:
            pixel_sigma: Standard deviation of noise in pixels.
                         Typical value: 0.5-1.0 pixels
            rng: NumPy random generator for reproducibility

        Returns:
            New ImagePoints with noisy 2D coordinates
        """
        df = self.image_points.df.copy()
        n_points = len(df)

        df["img_loc_x"] = df["img_loc_x"] + rng.normal(0, pixel_sigma, n_points)
        df["img_loc_y"] = df["img_loc_y"] + rng.normal(0, pixel_sigma, n_points)

        return ImagePoints(df)


def _create_camera_matrix(focal_length: float, image_size: tuple[int, int]) -> NDArray[np.float64]:
    """Create camera intrinsic matrix."""
    w, h = image_size
    cx, cy = w / 2, h / 2
    return np.array(
        [
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def _create_ring_cameras(
    n_cameras: int,
    ring_radius_mm: float,
    ring_height_mm: float,
    focal_length: float,
    image_size: tuple[int, int],
) -> CameraArray:
    """
    Create cameras in a ring formation looking toward the origin.

    Camera coordinate convention (OpenCV):
    - X: right (in image)
    - Y: down (in image)
    - Z: forward (optical axis, into the scene)

    World coordinate convention:
    - X: right
    - Y: forward (toward cameras when looking from above)
    - Z: up

    For a camera at position (x, y, z) looking at origin:
    1. Compute camera position in world coordinates
    2. Compute rotation matrix that aligns camera Z-axis with direction toward origin
    3. Convert to OpenCV rotation (camera-from-world transform)
    """
    cameras: dict[int, CameraData] = {}
    matrix = _create_camera_matrix(focal_length, image_size)
    # Standard 5-coefficient distortion model, all zeros (no distortion)
    distortions = np.zeros(5, dtype=np.float64)

    for i in range(n_cameras):
        angle = 2 * np.pi * i / n_cameras  # 0, 90, 180, 270 degrees

        # Camera position in world coordinates
        cam_x = ring_radius_mm * np.cos(angle)
        cam_y = ring_radius_mm * np.sin(angle)
        cam_z = ring_height_mm
        cam_position = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

        # Camera looks toward origin
        # Forward direction (camera Z in world): from camera toward origin
        forward = -cam_position / np.linalg.norm(cam_position)

        # Up hint (world Z)
        up_hint = np.array([0, 0, 1], dtype=np.float64)

        # Right direction (camera X in world): cross of forward and up
        right = np.cross(forward, up_hint)
        right = right / np.linalg.norm(right)

        # Actual up direction (camera Y in world, but Y is DOWN in OpenCV)
        # So we compute "down" direction
        down = np.cross(forward, right)
        down = down / np.linalg.norm(down)

        # Rotation matrix: world-to-camera transform
        # Columns are world basis vectors expressed in camera coordinates
        # R @ world_point = camera_point
        # R.T @ camera_point = world_point
        #
        # Camera axes in world coordinates:
        # - Camera X (right in image) = right
        # - Camera Y (down in image) = down
        # - Camera Z (forward/optical axis) = forward
        #
        # R_cam_from_world has rows = camera axes in world coordinates
        rotation = np.vstack([right, down, forward])  # (3, 3)

        # Translation: position of world origin in camera coordinates
        # t = -R @ cam_position_world
        translation = -rotation @ cam_position

        cameras[i] = CameraData(
            port=i,
            size=image_size,
            matrix=matrix.copy(),
            distortions=distortions.copy(),
            rotation=rotation,
            translation=translation,
        )

    return CameraArray(cameras=cameras)


def _generate_grid_trajectory(
    n_frames: int,
    grid_rows: int,
    grid_cols: int,
    grid_spacing_mm: float,
    trajectory_radius_mm: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """
    Generate a rigid grid following helical trajectory with slow tumble rotation.

    The grid (like charuco corners) maintains rigid structure and moves through
    space via SE(3) motion: helix translation + one full rotation over trajectory.

    Args:
        n_frames: Number of temporal frames
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        grid_spacing_mm: Distance between adjacent grid points
        trajectory_radius_mm: Radius of the helical trajectory
        rng: Random generator (unused but kept for API consistency)

    Returns:
        points: (n_frames * grid_rows * grid_cols, 3) world coordinates
        sync_indices: (n_frames * grid_rows * grid_cols,) frame indices
        point_ids: (n_frames * grid_rows * grid_cols,) point IDs (0 to rows*cols-1)
    """
    n_points = grid_rows * grid_cols

    # Define canonical grid centered at origin (XY plane)
    # Asymmetric grid makes orientation easier to perceive
    canonical_grid = np.zeros((n_points, 3), dtype=np.float64)
    for row in range(grid_rows):
        for col in range(grid_cols):
            point_idx = row * grid_cols + col
            # Center the grid at origin
            canonical_grid[point_idx, 0] = (col - (grid_cols - 1) / 2) * grid_spacing_mm
            canonical_grid[point_idx, 1] = (row - (grid_rows - 1) / 2) * grid_spacing_mm
            canonical_grid[point_idx, 2] = 0.0

    all_points = []
    all_sync_indices = []
    all_point_ids = []

    for frame_idx in range(n_frames):
        t = frame_idx / max(n_frames - 1, 1)  # 0 to 1

        # Trajectory center: helix around Z-axis (same as before)
        helix_angle = 2 * np.pi * t * 2  # Two full rotations of helix
        center_x = trajectory_radius_mm * np.cos(helix_angle)
        center_y = trajectory_radius_mm * np.sin(helix_angle)
        center_z = 200 * np.sin(np.pi * t)  # Oscillate up/down
        center = np.array([center_x, center_y, center_z])

        # Rotation: slow tumble (one full rotation over trajectory)
        # Rotate around an axis tilted 45 degrees from Z
        tumble_angle = 2 * np.pi * t  # One full rotation
        # Rotation axis: (1, 0, 1) normalized
        axis = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)

        # Rodrigues formula for rotation matrix
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        rotation = np.eye(3) + np.sin(tumble_angle) * K + (1 - np.cos(tumble_angle)) * (K @ K)

        # Transform grid: rotate then translate
        transformed_grid = (rotation @ canonical_grid.T).T + center

        for point_idx in range(n_points):
            all_points.append(transformed_grid[point_idx])
            all_sync_indices.append(frame_idx)
            all_point_ids.append(point_idx)

    return (
        np.array(all_points, dtype=np.float64),
        np.array(all_sync_indices, dtype=np.int64),
        np.array(all_point_ids, dtype=np.int64),
    )


def _project_points_to_cameras(
    world_points: NDArray[np.float64],  # (N, 3)
    sync_indices: NDArray[np.int64],  # (N,)
    point_ids: NDArray[np.int64],  # (N,)
    camera_array: CameraArray,
) -> pd.DataFrame:
    """
    Project 3D world points to 2D image coordinates for all cameras.

    Uses cv2.projectPoints with the camera's intrinsics and extrinsics.
    Only includes projections that fall within the image bounds.

    Returns:
        DataFrame with columns: sync_index, port, point_id, img_loc_x, img_loc_y,
                               obj_loc_x, obj_loc_y, obj_loc_z
    """
    rows = []

    for port, camera in camera_array.cameras.items():
        if camera.rotation is None or camera.translation is None:
            continue
        if camera.matrix is None or camera.distortions is None:
            continue

        # Project all points
        projected, _ = cv2.projectPoints(
            world_points.reshape(-1, 1, 3),
            camera.rotation,
            camera.translation,
            camera.matrix,
            camera.distortions,
        )
        projected = projected.reshape(-1, 2)  # (N, 2)

        # Filter points within image bounds
        w, h = camera.size
        for i in range(len(world_points)):
            x, y = projected[i]
            if 0 <= x < w and 0 <= y < h:
                rows.append(
                    {
                        "sync_index": int(sync_indices[i]),
                        "port": port,
                        "point_id": int(point_ids[i]),
                        "img_loc_x": float(x),
                        "img_loc_y": float(y),
                        "obj_loc_x": world_points[i, 0],
                        "obj_loc_y": world_points[i, 1],
                        "obj_loc_z": world_points[i, 2],
                        "frame_time": float(sync_indices[i]) / 30.0,  # Assume 30 fps
                    }
                )

    return pd.DataFrame(rows)


def create_four_camera_ring(
    focal_length: float = 800.0,
    image_size: tuple[int, int] = (1920, 1080),
    ring_radius_mm: float = 2000.0,
    ring_height_mm: float = 500.0,
    n_frames: int = 20,
    grid_rows: int = 5,
    grid_cols: int = 7,
    grid_spacing_mm: float = 50.0,
    trajectory_radius_mm: float = 200.0,
    seed: int = 42,
) -> SyntheticGroundTruth:
    """
    Create 4 cameras in a ring arrangement observing a moving rigid grid.

    Geometry:
    - 4 cameras at 90-degree intervals around a ring
    - Cameras at height ring_height_mm, looking toward origin
    - Rigid grid (like charuco corners) centered near origin, following a helical trajectory
    - Grid maintains rigid structure while undergoing SE(3) motion (translation + rotation)

    This geometry is well-conditioned for bundle adjustment:
    - Diverse viewpoints with good triangulation angles (~90 degrees between adjacent cameras)
    - Rigid grid structure matches calibration board constraints
    - All points visible from multiple cameras

    Args:
        focal_length: Camera focal length in pixels (same for fx, fy)
        image_size: (width, height) in pixels
        ring_radius_mm: Horizontal distance from cameras to origin
        ring_height_mm: Vertical height of cameras above the XY plane
        n_frames: Number of sync_indices (temporal frames)
        grid_rows: Number of rows in the calibration grid
        grid_cols: Number of columns in the calibration grid
        grid_spacing_mm: Distance between adjacent grid points in mm
        trajectory_radius_mm: Radius of the helical trajectory the grid follows
        seed: Random seed for reproducibility

    Returns:
        SyntheticGroundTruth with:
        - 4 cameras with perfect intrinsics and extrinsics
        - WorldPoints indexed by (sync_index, point_id)
        - ImagePoints with perfect 2D projections
    """
    rng = np.random.default_rng(seed)

    # Create cameras
    camera_array = _create_ring_cameras(
        n_cameras=4,
        ring_radius_mm=ring_radius_mm,
        ring_height_mm=ring_height_mm,
        focal_length=focal_length,
        image_size=image_size,
    )

    # Generate 3D points as rigid grid following trajectory
    points_3d, sync_indices, point_ids = _generate_grid_trajectory(
        n_frames=n_frames,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        grid_spacing_mm=grid_spacing_mm,
        trajectory_radius_mm=trajectory_radius_mm,
        rng=rng,
    )

    # Create WorldPoints
    world_df = pd.DataFrame(
        {
            "sync_index": sync_indices,
            "point_id": point_ids,
            "x_coord": points_3d[:, 0],
            "y_coord": points_3d[:, 1],
            "z_coord": points_3d[:, 2],
            "frame_time": sync_indices / 30.0,  # Assume 30 fps
        }
    )
    world_points = WorldPoints(world_df)

    # Project to 2D
    image_df = _project_points_to_cameras(
        world_points=points_3d,
        sync_indices=sync_indices,
        point_ids=point_ids,
        camera_array=camera_array,
    )
    image_points = ImagePoints(image_df)

    return SyntheticGroundTruth(
        cameras=camera_array,
        world_points=world_points,
        image_points=image_points,
    )
