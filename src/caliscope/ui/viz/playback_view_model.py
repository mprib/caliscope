import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        camera_array: CameraArray,
        world_points: WorldPoints | None = None,
        wireframe_segments: list[WireframeSegment] | None = None,
        fps: int = 30,
    ):
        self.world_points = world_points
        self.camera_array = camera_array
        self.wireframe_segments = wireframe_segments or []
        self.frame_rate = fps

        # Handle camera-only mode (no points)
        if world_points is None:
            self.all_point_ids = np.array([], dtype=np.int64)
            self.n_points = 0
            self.id_to_index: dict[int, int] = {}
            self._static_lines = np.empty((0, 3), dtype=np.int32)
            self._static_line_colors = np.empty((0, 3), dtype=np.float32)
            self._grouped_points: dict[Any, Any] = {}
            logger.info("PlaybackViewModel initialized in camera-only mode (no points).")
            return

        # 1. Establish the Canonical Map (The "Superset" of all points)
        # We find every unique point_id that appears in the entire recording.
        unique_ids = np.unique(world_points.df["point_id"].to_numpy())
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
        self._grouped_points = {idx: grp for idx, grp in world_points.df.groupby("sync_index")}

    @classmethod
    def from_xyz_csv(
        cls,
        xyz_path: str | Path,
        camera_array: CameraArray,
        wireframe_segments: list[WireframeSegment] | None = None,
        fps: int = 30,
    ) -> "PlaybackViewModel":
        """Create PlaybackViewModel from an xyz CSV file.

        Convenience factory for post-processing visualization where xyz data
        is loaded from disk rather than live triangulation.
        """
        world_points = WorldPoints.from_csv(xyz_path)
        return cls(
            camera_array=camera_array,
            world_points=world_points,
            wireframe_segments=wireframe_segments,
            fps=fps,
        )

    @classmethod
    def from_camera_array_only(cls, camera_array: CameraArray) -> "PlaybackViewModel":
        """Create a ViewModel with cameras only (no points).

        Used for preview mode before reconstruction — shows camera frustums
        without any tracked points.
        """
        return cls(camera_array=camera_array)

    @property
    def has_points(self) -> bool:
        """True if this contains point data, False if camera-only preview."""
        return self.world_points is not None

    @property
    def min_index(self) -> int:
        if self.world_points is None or self.world_points.min_index is None:
            return 0
        return self.world_points.min_index

    @property
    def max_index(self) -> int:
        if self.world_points is None or self.world_points.max_index is None:
            return 0  # Single frame for camera-only mode
        return self.world_points.max_index

    @property
    def valid_sync_indices(self) -> np.ndarray:
        """Return sorted array of sync indices that have point data.

        Used for slider navigation - the slider should only stop at frames
        that actually have data, not every frame in the min/max range.
        For sparse data (e.g., every 5th frame), this returns [0, 5, 10, ...]
        """
        if not self._grouped_points:
            return np.array([], dtype=np.int64)
        return np.sort(np.array(list(self._grouped_points.keys()), dtype=np.int64))

    def get_camera_geometry(self, scale: float = 0.0005) -> dict[str, Any] | None:
        """Pass-through to the static camera builder."""
        return build_camera_geometry(self.camera_array, scale=scale)

    def get_camera_positions(self) -> NDArray[np.float64] | None:
        """Get world positions of all cameras.

        Returns (n_cameras, 3) array of camera centers in world coordinates,
        or None if no cameras have extrinsics.

        Used by visualization to compute appropriate camera/view positioning.
        """
        positions = []
        for camera in self.camera_array.cameras.values():
            if camera.rotation is None or camera.translation is None:
                continue
            # Camera center in world: C = -R^T @ t
            center = -camera.rotation.T @ camera.translation
            positions.append(center)

        if not positions:
            return None

        return np.array(positions, dtype=np.float64)

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
            # p_ids from .values might be ExtensionArray - convert to numpy for isin
            valid_mask = np.isin(np.asarray(p_ids), self.all_point_ids)

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
