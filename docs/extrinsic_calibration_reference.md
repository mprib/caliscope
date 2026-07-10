# Extrinsic Calibration Reference

This page covers the mechanics behind extrinsic calibration: intrinsic refinement, the depth-ratio gate, skipping intrinsic calibration, and quality metrics.
For the step-by-step workflow, see [Extrinsic Calibration](extrinsic_calibration.md).

## Intrinsic Refinement

The Calibrate tab has a **"Refine camera intrinsics"** checkbox, on by default.
When checked, bundle adjustment re-estimates each camera's focal length and leading distortion coefficients (k1, k2) jointly with the camera poses, starting from whatever intrinsics you provided.
The principal point, tangential distortion (p1, p2), and k3 stay fixed.

Refinement usually improves the result.
Uncheck the box to lock your provided intrinsics when you trust a careful prior calibration more than the extrinsic footage.

Fisheye cameras are excluded from refinement.
The equidistant distortion model stays locked during bundle adjustment regardless of the checkbox.

When one or more cameras have no intrinsics at all (see [Skipping Intrinsic Calibration](#skipping-intrinsic-calibration)), the checkbox is forced on: the solver must recover those parameters.

## The Depth-Ratio Gate

Focal length and distance are coupled in the image.
Only depth variation separates them.

Before refining intrinsics, Caliscope measures each camera's near/far depth ratio.
**If any camera's ratio falls below 2.0, intrinsic refinement is disabled for the entire rig.**
Below that ratio, refining focal length drifts it and couples scale error into camera translation, which is worse than not refining at all.

Two consequences:

- Move the target toward and away from the cameras during recording. One camera with flat coverage gates the whole rig.
- The gate can silently override the checkbox. The log records per-camera depth ratios when the gate fires.

## Skipping Intrinsic Calibration

!!! warning "Experimental"
    This path is supported by the pipeline and passes synthetic tests, but has not been validated on real-world data to the same degree as the standard intrinsic-then-extrinsic workflow. The recommended path is to calibrate intrinsics first.

If a project has extrinsic videos but no intrinsic videos, Caliscope starts each camera from a rough guess (focal length from resolution, zero distortion) and refines focal length, k1, and k2 during bundle adjustment.
The principal point, tangential distortion, and k3 stay at their assumed values.

**Prerequisites:** depth variation (move the target toward and away from the cameras) and no fisheye cameras.
The [depth-ratio gate](#the-depth-ratio-gate) checks depth variation automatically.
Fisheye cameras require prior [intrinsic calibration](intrinsic_calibration.md).

For the file layout, see [Project Setup](project_setup.md#extrinsic-only-projects).

## Quality Metrics

After setting the origin, Caliscope computes volumetric scale accuracy:

**Pooled RMSE (mm):** The root-mean-square error of pairwise distances between reconstructed board corners, compared to the known board geometry. Measures how accurately the reconstruction preserves physical scale.

**Bias:** Consistently positive errors (reconstructed distances larger than known) suggest the entered board size is too small. Consistently negative errors suggest it is too large. This catches measurement errors in the board dimensions you entered during setup.
