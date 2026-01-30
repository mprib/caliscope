import tempfile
from pathlib import Path

import pytest

from caliscope.gui.geometry.wireframe import load_wireframe_config


def test_load_wireframe_config_valid():
    """Test successful loading of a valid wireframe TOML with points mapping."""
    toml_content = """
[points]
right_hip = 0
left_hip = 1
right_shoulder = 2
left_shoulder = 3

[pelvis]
color = "y"
points = ["right_hip", "left_hip"]

[right_flank]
color = "y"
points = ["right_hip", "right_shoulder"]

[left_flank]
color = "y"
points = ["left_hip", "left_shoulder"]
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        toml_path = Path(f.name)

    try:
        config = load_wireframe_config(toml_path)

        # Check mapping
        assert len(config.point_name_to_id) == 4
        assert config.point_name_to_id["right_hip"] == 0
        assert config.point_name_to_id["left_hip"] == 1

        # Check segments
        assert len(config.segments) == 3

        pelvis = next(s for s in config.segments if s.name == "pelvis")
        assert pelvis.point_a_id == 0  # right_hip
        assert pelvis.point_b_id == 1  # left_hip
        assert pelvis.color_rgb == (1.0, 1.0, 0.0)  # Yellow

    finally:
        toml_path.unlink()


def test_load_wireframe_config_missing_points_section():
    """Test error when TOML lacks [points] section."""
    toml_content = """
[pelvis]
color = "y"
points = ["right_hip", "left_hip"]
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        toml_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="must contain \\[points\\] section"):
            load_wireframe_config(toml_path)

    finally:
        toml_path.unlink()


def test_load_wireframe_config_default_color():
    """Test that default color (white) is applied when not specified."""
    toml_content = """
[points]
a = 0
b = 1

[segment_no_color]
points = ["a", "b"]
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        toml_path = Path(f.name)

    try:
        config = load_wireframe_config(toml_path)

        assert len(config.segments) == 1
        assert config.segments[0].color_rgb == (1.0, 1.0, 1.0)  # White

    finally:
        toml_path.unlink()


def test_load_wireframe_config_skips_missing_points():
    """Test that segments with missing point names are skipped."""
    toml_content = """
[points]
existing_a = 0
existing_b = 1

[good_segment]
color = "r"
points = ["existing_a", "existing_b"]

[bad_segment]
color = "g"
points = ["existing_a", "missing_point"]

[another_good_segment]
color = "b"
points = ["existing_b", "existing_a"]
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        toml_path = Path(f.name)

    try:
        config = load_wireframe_config(toml_path)

        # Should only load segments where both points exist
        assert len(config.segments) == 2

        segment_names = {seg.name for seg in config.segments}
        assert "good_segment" in segment_names
        assert "another_good_segment" in segment_names
        assert "bad_segment" not in segment_names  # Skipped

    finally:
        toml_path.unlink()


def test_load_wireframe_config_real_holistic_file():
    """Test loading the actual holistic wireframe TOML."""
    # Path to the holistic wireframe file in gui/geometry
    toml_path = (
        Path(__file__).parent.parent
        / "src"
        / "caliscope"
        / "gui"
        / "geometry"
        / "wireframes"
        / "holistic_wireframe.toml"
    )

    if not toml_path.exists():
        pytest.skip("Holistic wireframe TOML not found")

    config = load_wireframe_config(toml_path)

    # Should have mapping and segments
    assert len(config.point_name_to_id) > 0
    assert len(config.segments) > 0

    # Verify a specific segment exists
    segment_names = {seg.name for seg in config.segments}
    assert "left_arm" in segment_names

    # Verify color is in valid RGB range
    for segment in config.segments:
        assert len(segment.color_rgb) == 3
        assert all(0.0 <= c <= 1.0 for c in segment.color_rgb)
