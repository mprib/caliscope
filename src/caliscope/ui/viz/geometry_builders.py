"""
Pure functions to build renderable geometry from domain objects.
No UI imports - returns raw numpy arrays and connectivity data.
"""

import numpy as np
from numpy.typing import NDArray
from caliscope.cameras.camera_array import CameraArray


def build_camera_geometry(camera_array: CameraArray) -> dict[str, NDArray] | None:
    """
    Build static camera mesh geometry as pyramid frustums.
    Corrects the rotation math and aligns with OpenCV Y-down convention.
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

        # 1. Camera Intrinsics
        fx, fy = cam.matrix[0, 0], cam.matrix[1, 1]
        cx, cy = cam.matrix[0, 2], cam.matrix[1, 2]
        w, h = cam.size

        # Scale to reasonable size (normalize by focal length)
        # 0.0005 is a good baseline, but adjust if cameras look too large/small
        scale = 0.0005
        f_avg = (fx + fy) / 2 * scale

        # 2. Local Vertices (OpenCV convention: X-right, Y-down, Z-forward)
        # Apex is origin. Image corners at z = f_avg.
        # Top-Left (u=0, v=0) -> x = -cx, y = -cy
        # Bottom-Right (u=w, v=h) -> x = w-cx, y = h-cy
        verts_local = np.array(
            [
                [0, 0, 0],  # 0: Apex (Camera Center)
                [-cx * scale, -cy * scale, f_avg],  # 1: Top-Left
                [(w - cx) * scale, -cy * scale, f_avg],  # 2: Top-Right
                [(w - cx) * scale, (h - cy) * scale, f_avg],  # 3: Bottom-Right
                [-cx * scale, (h - cy) * scale, f_avg],  # 4: Bottom-Left
            ],
            dtype=np.float32,
        )

        # 3. Transform to world space
        # R_world_to_cam = cam.rotation
        # R_cam_to_world = cam.rotation.T
        # T_world = -R_cam_to_world @ cam.translation
        # Math: v_world = (R_cam_to_world @ v_local) + T_world
        # Row math: v_world = (v_local @ R_cam_to_world.T) + T_world
        # v_world = (v_local @ cam.rotation) + T_world

        R_cam_to_world = cam.rotation.T
        t_world = -R_cam_to_world @ cam.translation

        verts_world = (verts_local @ cam.rotation) + t_world

        # 4. Define faces (standard pyramid topology)
        # PyVista uses: [n_verts, v0, v1, v2, ...]
        faces = [
            [3, 0, 1, 2],  # Side: Apex to Top
            [3, 0, 2, 3],  # Side: Apex to Right
            [3, 0, 3, 4],  # Side: Apex to Bottom
            [3, 0, 4, 1],  # Side: Apex to Left
            [4, 1, 2, 3, 4],  # Base: Image Plane
        ]

        # Convert to flat array with vertex offsets for PyVista PolyData
        for face in faces:
            all_faces.append([face[0]] + [v + vertex_offset for v in face[1:]])

        all_vertices.append(verts_world)
        all_colors.append(np.tile(cam_color, (len(verts_world), 1)))

        # Add label position (slightly above the apex)
        label_pos = t_world + np.array([0, 0, 0.1], dtype=np.float32)
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
