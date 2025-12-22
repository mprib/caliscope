import pytest
from pathlib import Path


from caliscope.cameras.camera_array import CameraArray
from caliscope.ui.viz.playback_view_model import PlaybackViewModel
from caliscope.ui.viz.wireframe_loader import load_wireframe_config
import caliscope.persistence as persistence


def test_playback_view_model_with_real_data():
    """Integration test using real sample data from test fixtures."""
    # Path to test session
    session_path = Path(__file__).parent / "sessions" / "4_cam_recording"
    if not session_path.exists():
        pytest.skip("Test session data not found")

    # Load world points
    xyz_path = session_path / "recordings" / "recording_1" / "HOLISTIC" / "xyz_HOLISTIC.csv"
    if not xyz_path.exists():
        pytest.skip("XYZ data not found")

    world_points = persistence.load_world_points_csv(xyz_path)

    assert not world_points.df.empty

    # Load wireframe config (contains both mapping and segments)
    wireframe_path = (
        Path(__file__).parent.parent / "src" / "caliscope" / "ui" / "viz" / "wireframes" / "holistic_wireframe.toml"
    )
    if not wireframe_path.exists():
        pytest.skip("Holistic wireframe TOML not found")

    wireframe_config = load_wireframe_config(wireframe_path)

    # Verify mapping exists
    assert len(wireframe_config.point_name_to_id) > 0

    # Should have loaded some segments
    assert len(wireframe_config.segments) > 0

    # Create minimal camera array
    camera_array = CameraArray({})

    # Create ViewModel
    view_model = PlaybackViewModel(
        world_points=world_points,
        camera_array=camera_array,
        wireframe_segments=wireframe_config.segments,
    )

    # Test single frame geometry
    sync_index = world_points.df["sync_index"].iloc[0]
    point_geom = view_model.get_point_geometry(sync_index)
    assert point_geom is not None

    # Test wireframe geometry
    wireframe_geom = view_model.get_wireframe_geometry(sync_index)
    if wireframe_geom is not None:
        points, lines, colors = wireframe_geom
        assert points.shape[1] == 3
        assert lines.shape[1] == 3

    # Test "all points" mode
    all_points_geom = view_model.get_point_geometry(None)
    assert all_points_geom is not None

    # Wireframe should be None in all points mode
    all_wireframe_geom = view_model.get_wireframe_geometry(None)
    assert all_wireframe_geom is None
