"""
ViewModel for time-series playback visualization.
Encapsulates data slicing and geometry building. Immutable - no playback state.
"""

from dataclasses import dataclass

from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import WorldPoints
from caliscope.ui.viz.geometry_builders import build_point_geometry, build_wireframe_geometry, build_camera_geometry
from caliscope.ui.viz.wireframe_loader import WireframeSegment


@dataclass(frozen=True)  # Immutable - no state
class PlaybackViewModel:
    """
    ViewModel that holds full time-series data and provides render-ready geometry.

    This is a singleton per dataset - create one per motion trial.
    Widgets call methods with sync_index as parameter.

    Attributes:
        world_points: All 3D point data
        camera_array: Camera intrinsics/extrinsics (static)
        wireframe_segments: Optional wireframe segments (static)
    """

    world_points: WorldPoints
    camera_array: CameraArray
    wireframe_segments: list[WireframeSegment] | None

    def get_point_geometry(self, sync_index: int | None) -> tuple[NDArray, NDArray] | None:
        """
        Get point cloud geometry for specified sync_index.

        Args:
            sync_index: Frame index, or None for "all points" mode

        Returns:
            Tuple of (positions, colors) or None if no data
        """
        return build_point_geometry(self.world_points, sync_index)

    def get_wireframe_geometry(self, sync_index: int | None) -> tuple[NDArray, NDArray, NDArray] | None:
        """
        Get wireframe line geometry for specified sync_index.

        Args:
            sync_index: Frame index or None for all points

        Returns:
            Tuple of (points, lines, colors) or None if no wireframe/data
        """
        if not self.wireframe_segments:
            return None

        return build_wireframe_geometry(self.world_points, sync_index, self.wireframe_segments)

    def get_camera_geometry(self) -> dict | None:
        """
        Get static camera mesh geometry.

        Returns:
            Dictionary with vertices, faces, colors, labels or None
        """
        return build_camera_geometry(self.camera_array)
