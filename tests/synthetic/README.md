# Synthetic Testing Framework

Ground-truth-based testing for camera calibration bundle adjustment.

## Quick Start

```bash
# Run tests
uv run pytest tests/synthetic/ -v

# Visual verification (interactive 3-panel comparison)
uv run python tests/synthetic/test_extrinsic_calibration_synthetic.py
```

## Workflow

```
[1] Define scenes via factory functions (scene_factories.py)
        ↓
[2] Inspect visually via Explorer (view-only)
        ↓
[3] Export domain objects as fixtures (optional)
        ↓
[4] Load fixtures in tests (fast, deterministic)
```

## Architecture

```
src/caliscope/synthetic/
├── synthetic_scene.py       # SyntheticScene frozen dataclass
├── scene_factories.py       # Factory functions (default_ring_scene, etc.)
├── fixture_repository.py    # Fixture persistence (save/load)
├── se3_pose.py              # SE3 pose representation
├── trajectory.py            # Trajectory generators (orbital, linear, stationary)
├── calibration_object.py    # Rigid body with known geometry
├── camera_synthesizer.py    # Fluent builder for camera arrays
├── filter_config.py         # Visibility filtering (dropout, occlusion)
├── coverage.py              # Coverage matrix computation
└── explorer/                # Interactive GUI for exploring scenarios
    ├── presenter.py         # View-only presenter (accepts SyntheticScene)
    ├── explorer_tab.py
    └── widgets/

tests/synthetic/
├── test_extrinsic_calibration_synthetic.py  # Main calibration tests
├── primitives/              # Unit tests for domain primitives
└── fixtures/synthetic/      # Persisted test fixtures (gitignored or committed)
```

## Scene Factory Functions

Three pre-configured scenes in `scene_factories.py`:

```python
from caliscope.synthetic import default_ring_scene, sparse_coverage_scene, quick_test_scene

# Standard 4-camera ring, 20 frames, 5×7 grid
scene = default_ring_scene()

# 180° arc, larger radius for limited overlap
scene = sparse_coverage_scene()

# Minimal scene for fast tests (5 frames, 3×4 grid)
scene = quick_test_scene()
```

### Creating Custom Scenes

```python
from caliscope.synthetic import (
    SyntheticScene,
    CameraSynthesizer,
    CalibrationObject,
    Trajectory,
)

camera_array = (
    CameraSynthesizer()
    .add_ring(n=6, radius_mm=2500.0, height_mm=600.0)
    .build()
)

calibration_object = CalibrationObject.planar_grid(rows=7, cols=9, spacing_mm=40.0)
trajectory = Trajectory.orbital(n_frames=30, radius_mm=300.0, arc_extent_deg=360.0)

scene = SyntheticScene(
    camera_array=camera_array,
    calibration_object=calibration_object,
    trajectory=trajectory,
    pixel_noise_sigma=0.5,
    random_seed=42,
)
```

## Fixture Persistence

Save scenes as fixtures for fast, deterministic test loading:

```python
from caliscope.synthetic import save_fixture, load_fixture

# Save scene to tests/fixtures/synthetic/my_scenario/
save_fixture(scene, "my_scenario")

# Load fixture (returns SyntheticFixture with camera_array, image_points, world_points)
fixture = load_fixture("my_scenario")
```

**Fixture directory structure:**
```
tests/fixtures/synthetic/my_scenario/
├── camera_array.toml      # Ground truth cameras
├── world_points.csv       # Transformed object points per frame
├── image_points.csv       # Noisy projections
└── metadata.toml          # Scene parameters (noise, seed, counts)
```

## Explorer (Visual Verification)

The Explorer is a **view-only** tool for inspecting pre-defined scenes:

```python
from caliscope.synthetic import default_ring_scene
from caliscope.synthetic.explorer.presenter import ExplorerPresenter

scene = default_ring_scene()
presenter = ExplorerPresenter(task_manager, scene=scene)

# Replace scene
presenter.set_scene(another_scene)

# Run calibration pipeline
presenter.run_pipeline()
```

## Key Concepts

### Gauge Freedom

Bundle adjustment has 7 degrees of freedom (3 rotation, 3 translation, 1 scale) that can't be determined from images alone. We resolve this via `align_to_object()` which snaps results to the ground truth frame.

### Theory-Based Tolerances

Tolerances derive from covariance propagation, not arbitrary values:

```
Translation error ≈ GEOMETRY_FACTOR × pixel_sigma
```

Where `GEOMETRY_FACTOR ≈ 15-20` for typical setups. For `pixel_sigma=0.5`, expect max translation error of ~7-10mm.

### RMSE Convergence

If RMSE ≈ pixel_sigma, the optimizer converged to the noise floor, validating correctness.

## References

- Hartley & Zisserman, "Multiple View Geometry" Ch.18 (bundle adjustment theory)
- Triggs et al., "Bundle Adjustment - A Modern Synthesis" (covariance analysis)
