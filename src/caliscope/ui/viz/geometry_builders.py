"""
Pure functions to build renderable geometry from domain objects.
No UI imports - returns raw numpy arrays and connectivity data.
"""

import numpy as np
from numpy.typing import NDArray
from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import WorldPoints
from caliscope.ui.viz.wireframe_loader import WireframeSegment


def build_point_geometry(
    world_points: WorldPoints, sync_index: int | None
) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
    """
    Build point cloud geometry for a specific sync index.

    Args:
        world_points: Immutable container of 3D point data
        sync_index: Frame index, or None for "all points" mode

    Returns:
        Tuple of (positions, colors) as numpy arrays, or None if no data
        positions: (N, 3) array of xyz coordinates
        colors: (N, 3) array of RGB values in [0,1]
    """
    df = world_points.df

    if sync_index is not None:
        # Single frame mode
        frame_df = df[df["sync_index"] == sync_index]
    else:
        # All points mode - aggregate all points
        frame_df = df

    if frame_df.empty:
        return None

    positions = frame_df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)

    # Default to light gray points
    colors = np.full((len(positions), 3), 0.9, dtype=np.float32)

    return positions, colors


def build_wireframe_geometry(
    world_points: WorldPoints,
    sync_index: int | None,
    wireframe_segments: list[WireframeSegment],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]] | None:
    """
    Build wireframe line geometry.

    Args:
        world_points: 3D point data
        sync_index: Frame index or None for all points
        wireframe_segments: List of WireframeSegment from loader

    Returns:
        Tuple of (points, lines, colors) or None
        points: (V, 3) array of vertex positions
        lines: (L, 3) array where each row is [2, idx_a, idx_b]
        colors: (L, 3) array of RGB values per segment
    """
    # Skip wireframe in "all points" mode - too messy
    if sync_index is None:
        return None

    df = world_points.df[world_points.df["sync_index"] == sync_index]

    if df.empty or not wireframe_segments:
        return None

    # Fast lookup: point_id -> position
    point_positions = {
        int(row["point_id"]): np.array([row["x_coord"], row["y_coord"], row["z_coord"]], dtype=np.float32)
        for _, row in df.iterrows()
    }

    points: list[NDArray] = []
    lines: list[list[int]] = []
    colors: list[tuple[float, float, float]] = []

    vertex_idx = 0
    for segment in wireframe_segments:
        pos_a = point_positions.get(segment.point_a_id)
        pos_b = point_positions.get(segment.point_b_id)

        if pos_a is not None and pos_b is not None:
            points.append(pos_a)
            points.append(pos_b)
            lines.append([2, vertex_idx, vertex_idx + 1])
            colors.append(segment.color_rgb)
            vertex_idx += 2

    if not points:
        return None

    return (
        np.array(points, dtype=np.float32),
        np.array(lines, dtype=np.int32),
        np.array(colors, dtype=np.float32),
    )


def build_camera_geometry(camera_array: CameraArray) -> dict[str, NDArray] | None:
    """
    Build static camera mesh geometry as pyramid frustums.

    Returns:
        Dictionary with:
        - vertices: (V, 3) array of vertex positions
        - faces: (F, 4) array of quad indices (4 vertices per face)
        - colors: (V, 3) array of RGB values
        - labels: list of (position, text) tuples for camera labels
    """
    if not camera_array.all_extrinsics_calibrated():
        return None

    all_vertices = []
    all_faces = []
    all_colors = []
    labels = []
    vertex_offset = 0

    # Color scheme: green for cameras
    cam_color = np.array([0.2, 0.8, 0.2], dtype=np.float32)

    for port, cam in camera_array.cameras.items():
        if cam.rotation is None or cam.translation is None:
            continue

        # Build camera pyramid vertices in local space
        fx, fy = cam.matrix[0, 0], cam.matrix[1, 1]
        cx, cy = cam.matrix[0, 2], cam.matrix[1, 2]

        # Scale to reasonable size (normalize by focal length)
        scale = 0.01  # Adjust based on your scene scale
        f_avg = (fx + fy) / 2 * scale

        # Image plane corners in camera space
        w, h = cam.size
        verts_local = np.array(
            [
                [0, 0, 0],  # Apex (camera center)
                [(w - cx) * scale, (h - cy) * scale, f_avg],  # Top-right
                [(w - cx) * scale, -cy * scale, f_avg],  # Bottom-right
                [-cx * scale, -cy * scale, f_avg],  # Bottom-left
                [-cx * scale, (h - cy) * scale, f_avg],  # Top-left
            ],
            dtype=np.float32,
        )

        # Transform to world space
        R = cam.rotation.T  # Camera to world
        t = -R @ cam.translation  # Camera position in world

        verts_world = (verts_local @ R) + t

        # Define faces (4 triangular faces + 1 quad base)
        # PyVista uses: [n_verts, v0, v1, v2, ...]
        faces = [
            [3, 0, 1, 2],  # Triangle
            [3, 0, 2, 3],
            [3, 0, 3, 4],
            [3, 0, 4, 1],
            [4, 1, 2, 3, 4],  # Quad base
        ]

        # Convert to flat array with vertex offsets
        for face in faces:
            all_faces.append([face[0]] + [v + vertex_offset for v in face[1:]])

        all_vertices.append(verts_world)
        all_colors.append(np.tile(cam_color, (len(verts_world), 1)))

        # Add label position (above camera)
        label_pos = verts_world[0] + np.array([0, 0, 0.05], dtype=np.float32)
        labels.append((label_pos, f"Cam {port}"))

        vertex_offset += len(verts_world)

    if not all_vertices:
        return None

    return {
        "vertices": np.concatenate(all_vertices),
        "faces": np.array([item for sublist in all_faces for item in sublist], dtype=np.int32),
        "colors": np.concatenate(all_colors),
        "labels": labels,
    }
