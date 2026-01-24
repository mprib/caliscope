"""Tests for ScenarioConfig serialization and scene building."""

from __future__ import annotations

import numpy as np

from caliscope.synthetic.filter_config import FilterConfig
from caliscope.synthetic.scenario_config import (
    ScenarioConfig,
    default_ring_scenario,
    sparse_coverage_scenario,
)
from caliscope.synthetic.scene import SyntheticScene


def test_construction_with_all_parameters():
    """Can construct ScenarioConfig with all parameters."""
    filter_cfg = FilterConfig(
        dropped_cameras=(0,),
        killed_linkages=((0, 1),),
        dropped_frame_ranges=((5, 10),),
        random_dropout_fraction=0.1,
        random_seed=123,
    )

    config = ScenarioConfig(
        rig_type="ring",
        rig_params={"n_cameras": 4, "radius_mm": 2000.0, "height_mm": 500.0},
        trajectory_type="orbital",
        trajectory_params={
            "n_frames": 20,
            "radius_mm": 200.0,
            "arc_extent_deg": 360.0,
            "tumble_rate": 1.0,
        },
        object_type="planar_grid",
        object_params={"rows": 5, "cols": 7, "spacing_mm": 50.0},
        pixel_noise_sigma=0.8,
        filter_config=filter_cfg,
        random_seed=99,
        name="Test Config",
        description="A test scenario configuration",
    )

    assert config.rig_type == "ring"
    assert config.rig_params["n_cameras"] == 4
    assert config.trajectory_type == "orbital"
    assert config.trajectory_params["n_frames"] == 20
    assert config.object_type == "planar_grid"
    assert config.object_params["rows"] == 5
    assert config.pixel_noise_sigma == 0.8
    assert config.random_seed == 99
    assert config.name == "Test Config"
    assert config.description == "A test scenario configuration"
    assert config.filter_config.dropped_cameras == (0,)


def test_toml_roundtrip_preserves_values():
    """TOML serialization and deserialization recovers same values."""
    original = ScenarioConfig(
        rig_type="linear",
        rig_params={"n_cameras": 3, "spacing_mm": 500.0, "curvature": 0.5},
        trajectory_type="linear",
        trajectory_params={
            "n_frames": 10,
            "start": [0.0, -500.0, 100.0],
            "end": [0.0, 500.0, 100.0],
            "tumble_rate": 0.25,
        },
        object_type="planar_grid",
        object_params={"rows": 4, "cols": 6, "spacing_mm": 40.0, "origin": "center"},
        pixel_noise_sigma=1.2,
        filter_config=FilterConfig(dropped_cameras=(2,), random_dropout_fraction=0.05),
        random_seed=42,
        name="Linear Scenario",
        description="Testing linear trajectory",
    )

    # Serialize to TOML
    toml_str = original.to_toml()

    # Deserialize back
    recovered = ScenarioConfig.from_toml(toml_str)

    # Compare all fields
    assert recovered.rig_type == original.rig_type
    assert recovered.rig_params == original.rig_params
    assert recovered.trajectory_type == original.trajectory_type
    assert recovered.trajectory_params == original.trajectory_params
    assert recovered.object_type == original.object_type
    assert recovered.object_params == original.object_params
    assert recovered.pixel_noise_sigma == original.pixel_noise_sigma
    assert recovered.random_seed == original.random_seed
    assert recovered.name == original.name
    assert recovered.description == original.description
    assert recovered.filter_config.dropped_cameras == original.filter_config.dropped_cameras
    assert recovered.filter_config.random_dropout_fraction == original.filter_config.random_dropout_fraction


def test_toml_roundtrip_with_nested_rings():
    """TOML roundtrip works with nested_rings rig type."""
    original = ScenarioConfig(
        rig_type="nested_rings",
        rig_params={
            "inner_n": 3,
            "outer_n": 4,
            "inner_radius_mm": 1000.0,
            "outer_radius_mm": 2500.0,
            "inner_height_mm": 0.0,
            "outer_height_mm": 600.0,
        },
        trajectory_type="stationary",
        trajectory_params={"n_frames": 1},
        object_type="planar_grid",
        object_params={"rows": 5, "cols": 7, "spacing_mm": 50.0},
    )

    toml_str = original.to_toml()
    recovered = ScenarioConfig.from_toml(toml_str)

    assert recovered.rig_type == "nested_rings"
    assert recovered.rig_params["inner_n"] == 3
    assert recovered.rig_params["outer_n"] == 4


def test_build_scene_produces_valid_scene():
    """build_scene() produces a valid SyntheticScene."""
    config = default_ring_scenario()

    scene = config.build_scene()

    assert isinstance(scene, SyntheticScene)
    assert scene.n_cameras == 4
    assert scene.n_frames == 20
    assert scene.pixel_noise_sigma == 0.5
    assert scene.random_seed == 42

    # Scene should have valid derived data
    world_points = scene.world_points
    assert len(world_points.df) == 20 * 35  # 20 frames × 35 points (5×7 grid)

    image_points = scene.image_points_noisy
    assert len(image_points.df) > 0  # Should have some projections


def test_build_scene_with_linear_trajectory():
    """build_scene() works with linear trajectory (numpy array conversion)."""
    config = ScenarioConfig(
        rig_type="ring",
        rig_params={"n_cameras": 4, "radius_mm": 2000.0},
        trajectory_type="linear",
        trajectory_params={
            "n_frames": 15,
            "start": [-100.0, -100.0, 0.0],  # Lists (will be converted to arrays)
            "end": [100.0, 100.0, 0.0],
            "tumble_rate": 0.5,
        },
        object_type="planar_grid",
        object_params={"rows": 5, "cols": 7, "spacing_mm": 50.0},
    )

    scene = config.build_scene()

    assert scene.n_frames == 15
    # Verify trajectory was built correctly
    assert len(scene.trajectory) == 15


def test_build_scene_with_stationary_trajectory():
    """build_scene() works with stationary trajectory."""
    config = ScenarioConfig(
        rig_type="ring",
        rig_params={"n_cameras": 4, "radius_mm": 2000.0},
        trajectory_type="stationary",
        trajectory_params={"n_frames": 8},
        object_type="planar_grid",
        object_params={"rows": 5, "cols": 7, "spacing_mm": 50.0},
    )

    scene = config.build_scene()

    assert scene.n_frames == 8
    # All trajectory poses should be identical (stationary)
    for i in range(1, len(scene.trajectory)):
        pose_0 = scene.trajectory[0]
        pose_i = scene.trajectory[i]
        assert np.allclose(pose_0.translation, pose_i.translation)
        assert np.allclose(pose_0.rotation, pose_i.rotation)


def test_factory_default_ring_scenario():
    """default_ring_scenario() produces valid config."""
    config = default_ring_scenario()

    assert config.rig_type == "ring"
    assert config.rig_params["n_cameras"] == 4
    assert config.trajectory_type == "orbital"
    assert config.trajectory_params["n_frames"] == 20
    assert config.name == "Default Ring"

    # Should build successfully
    scene = config.build_scene()
    assert scene.n_cameras == 4


def test_factory_sparse_coverage_scenario():
    """sparse_coverage_scenario() produces valid config."""
    config = sparse_coverage_scenario()

    assert config.rig_type == "ring"
    assert config.trajectory_params["arc_extent_deg"] == 180.0  # Half orbit
    assert config.name == "Sparse Coverage"

    # Should build successfully
    scene = config.build_scene()
    assert scene.n_cameras == 4


def test_build_scene_all_rig_types():
    """build_scene() works with all rig types."""
    rig_configs = [
        ("ring", {"n_cameras": 4, "radius_mm": 2000.0}),
        ("linear", {"n_cameras": 3, "spacing_mm": 500.0}),
        (
            "nested_rings",
            {
                "inner_n": 3,
                "outer_n": 4,
                "inner_radius_mm": 1000.0,
                "outer_radius_mm": 2500.0,
            },
        ),
    ]

    for rig_type, rig_params in rig_configs:
        config = ScenarioConfig(
            rig_type=rig_type,  # type: ignore[arg-type]
            rig_params=rig_params,
            trajectory_type="orbital",
            trajectory_params={"n_frames": 10, "radius_mm": 200.0},
            object_type="planar_grid",
            object_params={"rows": 5, "cols": 7, "spacing_mm": 50.0},
        )

        scene = config.build_scene()
        assert isinstance(scene, SyntheticScene)


def test_build_scene_all_trajectory_types():
    """build_scene() works with all trajectory types."""
    trajectory_configs = [
        ("orbital", {"n_frames": 10, "radius_mm": 200.0}),
        (
            "linear",
            {
                "n_frames": 10,
                "start": [0.0, -200.0, 0.0],
                "end": [0.0, 200.0, 0.0],
            },
        ),
        ("stationary", {"n_frames": 5}),
    ]

    for traj_type, traj_params in trajectory_configs:
        config = ScenarioConfig(
            rig_type="ring",
            rig_params={"n_cameras": 4, "radius_mm": 2000.0},
            trajectory_type=traj_type,  # type: ignore[arg-type]
            trajectory_params=traj_params,
            object_type="planar_grid",
            object_params={"rows": 5, "cols": 7, "spacing_mm": 50.0},
        )

        scene = config.build_scene()
        assert isinstance(scene, SyntheticScene)


def test_toml_serialization_is_human_readable():
    """TOML output has clear section structure."""
    config = default_ring_scenario()
    toml_str = config.to_toml()

    # Should contain section headers
    assert "[metadata]" in toml_str
    assert "[camera_rig]" in toml_str
    assert "[trajectory]" in toml_str
    assert "[calibration_object]" in toml_str
    assert "[noise]" in toml_str
    assert "[filter]" in toml_str

    # Should contain type fields
    assert 'type = "ring"' in toml_str
    assert 'type = "orbital"' in toml_str
    assert 'type = "planar_grid"' in toml_str


def test_build_scene_respects_pixel_noise_sigma():
    """build_scene() passes pixel_noise_sigma to SyntheticScene."""
    config = ScenarioConfig(
        rig_type="ring",
        rig_params={"n_cameras": 4, "radius_mm": 2000.0},
        trajectory_type="orbital",
        trajectory_params={"n_frames": 10, "radius_mm": 200.0},
        object_type="planar_grid",
        object_params={"rows": 5, "cols": 7, "spacing_mm": 50.0},
        pixel_noise_sigma=2.5,
        random_seed=999,
    )

    scene = config.build_scene()

    assert scene.pixel_noise_sigma == 2.5
    assert scene.random_seed == 999


if __name__ == "__main__":
    """Debug harness for running tests and inspecting outputs."""
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Running test suite...")

    # Run all tests
    test_construction_with_all_parameters()
    print("✓ test_construction_with_all_parameters")

    test_toml_roundtrip_preserves_values()
    print("✓ test_toml_roundtrip_preserves_values")

    test_toml_roundtrip_with_nested_rings()
    print("✓ test_toml_roundtrip_with_nested_rings")

    test_build_scene_produces_valid_scene()
    print("✓ test_build_scene_produces_valid_scene")

    test_build_scene_with_linear_trajectory()
    print("✓ test_build_scene_with_linear_trajectory")

    test_build_scene_with_stationary_trajectory()
    print("✓ test_build_scene_with_stationary_trajectory")

    test_factory_default_ring_scenario()
    print("✓ test_factory_default_ring_scenario")

    test_factory_sparse_coverage_scenario()
    print("✓ test_factory_sparse_coverage_scenario")

    test_build_scene_all_rig_types()
    print("✓ test_build_scene_all_rig_types")

    test_build_scene_all_trajectory_types()
    print("✓ test_build_scene_all_trajectory_types")

    test_toml_serialization_is_human_readable()
    print("✓ test_toml_serialization_is_human_readable")

    test_build_scene_respects_pixel_noise_sigma()
    print("✓ test_build_scene_respects_pixel_noise_sigma")

    print("\nAll tests passed!")

    # Save a sample TOML for inspection
    config = default_ring_scenario()
    toml_path = debug_dir / "sample_scenario.toml"
    toml_path.write_text(config.to_toml())
    print(f"\nSample TOML saved to: {toml_path}")

    # Build a scene and inspect
    scene = config.build_scene()
    print(f"\nBuilt scene: {scene.n_cameras} cameras, {scene.n_frames} frames")
    print(f"World points: {len(scene.world_points)} total")
    print(f"Image points (noisy): {len(scene.image_points_noisy)} observations")
