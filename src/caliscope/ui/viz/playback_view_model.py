import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import WorldPoints
from caliscope.ui.viz.geometry_builders import build_camera_geometry
from caliscope.ui.viz.wireframe_loader import WireframeSegment

logger = logging.getLogger(__name__)


@dataclass
class FrameGeometry:
    """Holds the raw buffers for a single frame, ready for the GPU."""

    points: NDArray[np.float32]  # (N, 3) float32, includes NaNs for missing points
    colors: NDArray[np.float32]  # (N, 3) float32, RGB


class PlaybackViewModel:
    def __init__(
        self,
        world_points: WorldPoints,
        camera_array: CameraArray,
        wireframe_segments: list[WireframeSegment] | None = None,
        fps: int = 30,
    ):
        self.world_points = world_points
        self.camera_array = camera_array
        self.wireframe_segments = wireframe_segments or []
        self.frame_rate = fps

        # 1. Establish the Canonical Map (The "Superset" of all points)
        # We find every unique point_id that appears in the entire recording.
        unique_ids = np.unique(self.world_points.df["point_id"].values)
        unique_ids.sort()

        self.all_point_ids = unique_ids
        self.n_points = len(unique_ids)

        # Map: Point ID -> Buffer Index (0 to N-1)
        self.id_to_index = {pid: i for i, pid in enumerate(unique_ids)}

        logger.info(f"PlaybackViewModel initialized with {self.n_points} unique points.")

        # 2. Pre-compute Static Wireframe Topology
        # This converts point IDs to buffer indices.
        # Result is (n_lines, 3) array: [2, index_A, index_B] per row.
        self._static_lines, self._static_line_colors = self._build_static_topology()

        # 3. Pre-group data for fast lookup during playback
        # We group by sync_index so we don't have to filter the huge dataframe every frame.
        self._grouped_points = {idx: grp for idx, grp in self.world_points.df.groupby("sync_index")}

    @property
    def min_index(self) -> int:
        return self.world_points.min_index if self.world_points.min_index is not None else 0

    @property
    def max_index(self) -> int:
        return self.world_points.max_index if self.world_points.max_index is not None else 100

    def get_camera_geometry(self) -> dict[str, NDArray] | None:
        """Pass-through to the static camera builder."""
        return build_camera_geometry(self.camera_array)

    def get_static_wireframe_data(self) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """
        Returns the static connectivity data for the wireframe.

        Returns:
            lines: (L, 3) int32 array. Format: [2, index_A, index_B]
            colors: (L, 3) float32 array. RGB colors for each line segment.
        """
        return self._static_lines, self._static_line_colors

    def get_frame_geometry(self, sync_index: int) -> FrameGeometry:
        """
        Returns the point coordinates for a specific frame.

        Crucially, this returns a FIXED SIZE array (N, 3).
        Points missing in this frame are filled with NaN.
        """
        # Initialize with NaN (invisible in VTK)
        # Shape is (N, 3)
        points_buffer = np.full((self.n_points, 3), np.nan, dtype=np.float32)

        # Default color: Light Grey
        colors_buffer = np.full((self.n_points, 3), 0.8, dtype=np.float32)

        if sync_index in self._grouped_points:
            frame_df = self._grouped_points[sync_index]

            # Extract raw data
            p_ids = frame_df["point_id"].values
            coords = frame_df[["x_coord", "y_coord", "z_coord"]].values.astype(np.float32)

            # Vectorized Scatter
            # We map the frame's point_ids to their canonical buffer indices
            # Note: We filter out any IDs that might not be in our canonical set (safety)
            valid_mask = np.isin(p_ids, self.all_point_ids)

            if np.any(valid_mask):
                p_ids = p_ids[valid_mask]
                coords = coords[valid_mask]

                # Look up indices
                indices = [self.id_to_index[pid] for pid in p_ids]

                # Scatter into the buffer
                points_buffer[indices] = coords

                # Here you could also scatter dynamic colors (e.g. confidence) if available
                # colors_buffer[indices] = ...

        return FrameGeometry(points=points_buffer, colors=colors_buffer)

    def _build_static_topology(self) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """
        Converts the logical wireframe (Point A -> Point B) into
        buffer indices (Index 5 -> Index 12).
        """
        lines = []
        colors = []

        for segment in self.wireframe_segments:
            # Only create lines if BOTH points exist in our dataset
            if segment.point_a_id in self.id_to_index and segment.point_b_id in self.id_to_index:
                idx_a = self.id_to_index[segment.point_a_id]
                idx_b = self.id_to_index[segment.point_b_id]

                # PyVista line format: [num_points, idx1, idx2]
                lines.append([2, idx_a, idx_b])
                colors.append(segment.color_rgb)

        if not lines:
            # Return empty arrays if no wireframe
            return (
                np.empty((0, 3), dtype=np.int32),
                np.empty((0, 3), dtype=np.float32),
            )

        return (
            np.array(lines, dtype=np.int32),
            np.array(colors, dtype=np.float32),
        )
