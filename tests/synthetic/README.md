# Synthetic Testing Framework

Ground-truth-based testing for camera calibration bundle adjustment.

## Quick Start

```bash
# Run tests
uv run pytest tests/synthetic/ -v

# Visual verification (interactive 3-panel comparison)
uv run python tests/synthetic/test_extrinsic_calibration_synthetic.py
```

## Architecture

Synthetic testing is split into **domain primitives** (production code in `src/caliscope/synthetic/`) and **test utilities** (in `tests/synthetic/`).

```
src/caliscope/synthetic/     # Domain primitives (production code)
├── se3_pose.py              # SE3 pose representation
├── trajectory.py            # Trajectory generators (orbital, linear, stationary)
├── calibration_object.py    # Rigid body with known geometry
├── camera_rigs.py           # Camera arrangement factories (ring, linear, nested)
├── scene.py                 # SyntheticScene - combines cameras + object + trajectory
├── scenario_config.py       # TOML-serializable configuration
├── filter_config.py         # Visibility filtering (dropout, occlusion)
├── coverage.py              # Coverage matrix computation
└── explorer/                # Interactive GUI for exploring scenarios
    ├── presenter.py
    ├── explorer_tab.py
    └── widgets/
        ├── coverage_heatmap.py
        └── storyboard_view.py

tests/synthetic/
├── assertions.py            # Pose error comparison helpers
├── test_cases.py            # Factory: scene → noisy input → optimized
├── test_extrinsic_calibration_synthetic.py  # Actual pytest tests
├── primitives/              # Unit tests for synthetic domain primitives
│   ├── test_se3_pose.py
│   ├── test_trajectory.py
│   ├── test_calibration_object.py
│   ├── test_camera_rigs.py
│   ├── test_scene.py
│   ├── test_scenario_config.py
│   ├── test_filter_config.py
│   ├── test_coverage.py
│   ├── test_coverage_heatmap.py
│   ├── test_explorer_presenter.py
│   └── test_explorer_tab.py
└── widgets/
    └── storyboard.py        # 3-panel visual comparison widget
```

## How It Works

1. **Generate ground truth** - Use `ScenarioConfig` to build a `SyntheticScene` with perfect cameras, calibration object, and trajectory
2. **Add noise** - Perturb camera poses (rotation/translation), add Gaussian noise to 2D observations
3. **Run optimization** - Bundle adjustment via `PointDataBundle.optimize()`
4. **Compare** - Measure how close optimized result is to ground truth

## Creating Test Scenarios

The `ScenarioConfig` class provides a TOML-serializable way to define synthetic scenarios. It can be used in tests or exported/imported for reproducibility.

### Using Presets

```python
from caliscope.synthetic.scenario_config import (
    default_ring_scenario,
    sparse_coverage_scenario,
    occluded_camera_scenario,
)

# Use a preset
config = default_ring_scenario()
scene = config.build_scene()

# Access generated data
world_points = scene.world_points  # WorldPoints DataFrame
image_points = scene.image_points_noisy  # ImagePoints with noise
cameras = scene.camera_array  # Fully posed cameras
```

### Custom Configuration

```python
from caliscope.synthetic.scenario_config import ScenarioConfig

config = ScenarioConfig(
    # Camera rig
    rig_type="ring",
    rig_params={"n_cameras": 4, "radius_mm": 2000.0, "height_mm": 500.0},

    # Trajectory
    trajectory_type="orbital",
    trajectory_params={
        "n_frames": 20,
        "radius_mm": 200.0,
        "arc_extent_deg": 360.0,
        "tumble_rate": 1.0,
    },

    # Calibration object
    object_type="planar_grid",
    object_params={"rows": 5, "cols": 7, "spacing_mm": 50.0},

    # Noise and filtering
    pixel_noise_sigma=0.5,
    random_seed=42,

    # Metadata
    name="Custom Scenario",
    description="Brief description of what this tests",
)

scene = config.build_scene()
```

### TOML Serialization

```python
# Export to file
toml_str = config.to_toml()
with open("scenario.toml", "w") as f:
    f.write(toml_str)

# Load from file
with open("scenario.toml", "r") as f:
    loaded_config = ScenarioConfig.from_toml(f.read())

scene = loaded_config.build_scene()
```

## Key Concepts

### Gauge Freedom

Bundle adjustment has 7 degrees of freedom that can't be determined from images:
- 3 rotation, 3 translation, 1 scale

We resolve this via `align_to_object()` which uses known object coordinates
(`obj_loc_x/y/z`) to snap the result back to the ground truth frame.

**All cameras are perturbed** - there's no "gauge reference" camera.

### Theory-Based Tolerances

Tolerances are derived from covariance propagation, not arbitrary:

```
Translation error ≈ GEOMETRY_FACTOR × pixel_sigma
```

Where `GEOMETRY_FACTOR ≈ 15-20` for our setup (derived from σ_trans ≈ Z²/(f×B) × σ_pixel).

For `pixel_sigma=0.5`, expect max translation error of ~7-10mm.

### RMSE Convergence

If RMSE ≈ pixel_sigma, the optimizer converged to the noise floor.
This validates the optimizer is working correctly.

## Extending

### Adding New Test Scenarios

The easiest way is to create a new preset in `scenario_config.py`:

```python
def my_new_scenario() -> ScenarioConfig:
    """Description of what this tests."""
    return ScenarioConfig(
        rig_type="ring",
        rig_params={"n_cameras": 6, "radius_mm": 3000.0, "height_mm": 800.0},
        trajectory_type="linear",
        trajectory_params={"n_frames": 10, "start": [0, 0, 0], "end": [500, 0, 0]},
        object_type="planar_grid",
        object_params={"rows": 7, "cols": 9, "spacing_mm": 40.0},
        name="My Scenario",
        description="Tests X under Y conditions",
    )
```

Then use it in tests:

```python
from tests.synthetic.test_cases import create_extrinsic_calibration_test_case

def test_my_scenario():
    # Override the default scenario by modifying config
    test_case = create_extrinsic_calibration_test_case(
        seed=42,
        n_frames=10,  # Override trajectory length
    )
    # Or build directly from your config
    scene = my_new_scenario().build_scene()
    # Use scene.world_points, scene.image_points_noisy, etc.
```

### Adding New Trajectory Types

1. Add generator method to `Trajectory` class in `src/caliscope/synthetic/trajectory.py`
2. Update `ScenarioConfig.build_scene()` to handle new type
3. Add unit tests in `tests/synthetic/primitives/test_trajectory.py`

### Adding New Camera Rigs

1. Add factory function to `src/caliscope/synthetic/camera_rigs.py`
2. Update `ScenarioConfig.build_scene()` to handle new rig type
3. Add unit tests in `tests/synthetic/primitives/test_camera_rigs.py`

### Adding New Assertions

If you need new error metrics beyond pose error and RMSE:

1. Add functions to `tests/synthetic/assertions.py`
2. Add computed properties to `ExtrinsicCalibrationTestCase` if appropriate

## Visual Verification

The storyboard widget shows three panels side-by-side:
- **Ground Truth** - Perfect cameras and points
- **Noisy Input** - Perturbed cameras, noisy observations
- **Optimized** - Result of bundle adjustment

All panels are synchronized (rotate one, all rotate). Error metrics shown below.

## References

- Hartley & Zisserman, "Multiple View Geometry" Ch.18 (bundle adjustment theory)
- Triggs et al., "Bundle Adjustment - A Modern Synthesis" (covariance analysis)
