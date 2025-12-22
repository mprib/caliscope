import numpy as np
import pandas as pd

from caliscope.core.point_data import WorldPoints
from caliscope.ui.viz.geometry_builders import build_point_geometry, build_wireframe_geometry
from caliscope.ui.viz.wireframe_loader import WireframeSegment


def test_build_point_geometry_single_frame():
    """Test point geometry for a specific sync index."""
    df = pd.DataFrame(
        {
            "sync_index": [0, 0, 1, 1],
            "point_id": [0, 1, 0, 1],
            "x_coord": [1.0, 2.0, 1.1, 2.1],
            "y_coord": [0.0, 0.0, 0.1, 0.1],
            "z_coord": [0.0, 0.0, 0.1, 0.1],
        }
    )
    world_points = WorldPoints(df)

    result = build_point_geometry(world_points, sync_index=0)
    assert result is not None

    positions, colors = result
    assert positions.shape == (2, 3)
    assert colors.shape == (2, 3)
    assert np.allclose(positions[0], [1.0, 0.0, 0.0])
    assert np.allclose(positions[1], [2.0, 0.0, 0.0])


def test_build_point_geometry_all_points():
    """Test point geometry for all points mode."""
    df = pd.DataFrame(
        {
            "sync_index": [0, 0, 1, 1],
            "point_id": [0, 1, 0, 1],
            "x_coord": [1.0, 2.0, 1.1, 2.1],
            "y_coord": [0.0, 0.0, 0.1, 0.1],
            "z_coord": [0.0, 0.0, 0.1, 0.1],
        }
    )
    world_points = WorldPoints(df)

    result = build_point_geometry(world_points, sync_index=None)
    assert result is not None

    positions, colors = result
    assert positions.shape == (4, 3)


def test_build_wireframe_geometry():
    """Test wireframe geometry building."""
    df = pd.DataFrame(
        {
            "sync_index": [0, 0, 1, 1],
            "point_id": [0, 1, 0, 1],
            "x_coord": [0.0, 1.0, 0.1, 1.1],
            "y_coord": [0.0, 0.0, 0.1, 0.1],
            "z_coord": [0.0, 0.0, 0.1, 0.1],
        }
    )
    world_points = WorldPoints(df)

    segments = [
        WireframeSegment("segment_1", point_a_id=0, point_b_id=1, color_rgb=(1.0, 0.0, 0.0)),
    ]

    result = build_wireframe_geometry(world_points, sync_index=0, wireframe_segments=segments)
    assert result is not None

    points, lines, colors = result
    assert points.shape == (2, 3)  # Two vertices for one segment
    assert lines.shape == (1, 3)  # One line: [2, 0, 1]
    assert colors.shape == (1, 3)  # One color per segment


def test_build_wireframe_geometry_missing_points():
    """Test wireframe skips segments when points are missing."""
    df = pd.DataFrame(
        {
            "sync_index": [0, 0],
            "point_id": [0, 1],
            "x_coord": [0.0, 1.0],
            "y_coord": [0.0, 0.0],
            "z_coord": [0.0, 0.0],
        }
    )
    world_points = WorldPoints(df)

    segments = [
        WireframeSegment("good", point_a_id=0, point_b_id=1, color_rgb=(1.0, 0.0, 0.0)),
        WireframeSegment("bad", point_a_id=0, point_b_id=999, color_rgb=(0.0, 1.0, 0.0)),
    ]

    result = build_wireframe_geometry(world_points, sync_index=0, wireframe_segments=segments)
    assert result is not None

    points, lines, colors = result
    assert points.shape == (2, 3)  # Only the good segment
    assert lines.shape == (1, 3)
    assert colors.shape == (1, 3)


if __name__ == "__main__":
    test_build_wireframe_geometry_missing_points()
    test_build_point_geometry_all_points()
    test_build_point_geometry_single_frame()
    test_build_wireframe_geometry()
