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

```
tests/synthetic/
├── synthetic_scene.py       # Creates perfect cameras, points, projections
├── test_cases.py            # Factory: ground truth → noisy input → optimized
├── assertions.py            # Pose error comparison helpers
├── test_extrinsic_calibration_synthetic.py  # Actual pytest tests
└── widgets/
    └── storyboard.py        # 3-panel visual comparison widget
```

## How It Works

1. **Generate ground truth** - 4 cameras in a ring, 5x7 grid moving through space
2. **Add noise** - Perturb camera poses, add Gaussian noise to 2D observations
3. **Run optimization** - Bundle adjustment via `PointDataBundle.optimize()`
4. **Compare** - Measure how close optimized result is to ground truth

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

To add new test scenarios:

1. Add generation functions to `synthetic_scene.py` if needed
2. Add factory function in `test_cases.py`
3. Add assertions in `assertions.py` if needed
4. Write tests in a new `test_*.py` file

## Visual Verification

The storyboard widget shows three panels side-by-side:
- **Ground Truth** - Perfect cameras and points
- **Noisy Input** - Perturbed cameras, noisy observations
- **Optimized** - Result of bundle adjustment

All panels are synchronized (rotate one, all rotate). Error metrics shown below.

## References

- Hartley & Zisserman, "Multiple View Geometry" Ch.18 (bundle adjustment theory)
- Triggs et al., "Bundle Adjustment - A Modern Synthesis" (covariance analysis)
