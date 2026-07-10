# Scripting API

Caliscope's calibration pipeline is available as Python functions.
You can run intrinsic and extrinsic calibration from a script without the GUI.

The standard install includes the API:

```bash
uv pip install caliscope        # calibration library + scripting API
uv pip install caliscope[gui]   # adds desktop app, 3D visualization, and pose tracking
```

Everything below uses the `caliscope.api` module.
A complete working script is in `scripts/demo_api.py`.

## Imports

```python
from pathlib import Path
from caliscope.api import (
    Charuco, CharucoTracker, CameraArray, CaptureVolume,
    extract_image_points, extract_image_points_multicam,
    calibrate_intrinsics, calibrate_extrinsics,
)
from caliscope.core.constraints import ConstraintSet
from caliscope.reporting import (
    print_intrinsic_report, print_extrinsic_report, print_camera_pair_coverage,
)
```

`caliscope.api` re-exports the domain classes and calibration functions.
`caliscope.reporting` provides Rich terminal output for quality inspection.

## Step 1: Define the calibration target

Create a Charuco board matching the physical board you recorded:

```python
charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=3.0)
tracker = CharucoTracker(charuco)
```

`square_size_cm` sets the scale of all calibrated coordinates.
A 3.0 cm square produces corners spaced 0.03 m apart in object space.
Measure your printed board and use the actual size.

## Step 2: Build a camera array from video metadata

```python
intrinsic_videos = {0: Path("intrinsic/cam_0.mp4"), 1: Path("intrinsic/cam_1.mp4")}
cameras = CameraArray.from_video_metadata(intrinsic_videos)
```

This reads resolution and frame rate from each video and creates uncalibrated `CameraData` entries.
The dictionary keys are camera IDs, which must stay consistent across intrinsic and extrinsic videos.

## Step 3: Intrinsic calibration

For each camera, extract charuco corners from the intrinsic video, then solve for the camera matrix and distortion coefficients:

```python
for cam_id, video in intrinsic_videos.items():
    points = extract_image_points(video, cam_id, tracker, frame_step=5)
    output = calibrate_intrinsics(points, cameras[cam_id])
    cameras[cam_id] = output.camera

    print_intrinsic_report(output)
```

`frame_step=5` processes every 5th frame.
Intrinsic calibration needs roughly 30 diverse frames, so skipping frames saves time without sacrificing quality.

`extract_image_points` shows a Rich progress bar by default.
Pass `progress=None` to suppress it.

`calibrate_intrinsics` returns an `IntrinsicCalibrationOutput` containing the calibrated `CameraData` and an `IntrinsicCalibrationReport` with RMSE and coverage metrics.
`print_intrinsic_report` prints these with color-coded quality badges.

This whole step is optional.
The extrinsic pipeline can recover intrinsics on its own; see [Calibrating without intrinsics](#calibrating-without-intrinsics) below.

## Step 4: Extract time-aligned points for extrinsic calibration

```python
extrinsic_videos = {0: Path("extrinsic/cam_0.mp4"), 1: Path("extrinsic/cam_1.mp4")}
ext_points = extract_image_points_multicam(extrinsic_videos, tracker)
```

This function reads all videos concurrently (one thread per camera), aligns frames by timestamp, and runs the tracker on each time-aligned moment.
The result is a single `ImagePoints` DataFrame with observations from all cameras.

If your cameras were not hardware-synchronized, pass a timestamps file:

```python
ext_points = extract_image_points_multicam(extrinsic_videos, tracker, timestamps="timestamps.csv")
```

`frame_step` works on time-aligned moments, not raw frames.
`frame_step=10` processes every 10th synchronized moment.

### Check camera pair coverage

Before calibrating, verify that camera pairs share enough observations:

```python
print_camera_pair_coverage(ext_points)
```

This prints a lower-triangle grid of shared observation counts and flags structural problems (e.g., a camera pair with zero overlap).

## Step 5: Calibrate

`calibrate_extrinsics` runs the same pipeline as the GUI's Calibrate button: bootstrap camera poses from pairwise PnP, robust bundle adjustment, outlier filtering, and a final re-optimization.

```python
constraints = ConstraintSet.from_charuco(charuco)
result = calibrate_extrinsics(ext_points, cameras, constraints)
volume = result.capture_volume
```

The constraints feed the board's known corner geometry into bundle adjustment as rigidity information, which stabilizes the solution and locks world scale.
Build them from the same board definition you used for tracking.

Two keyword arguments matter most:

- `refine_intrinsics` (default `True`): re-estimate each camera's focal length and leading distortion jointly with the poses.
  Refinement is subject to the [depth-ratio gate](extrinsic_calibration.md#the-depth-ratio-gate): if any camera saw the target over too narrow a depth range, refinement is disabled for the whole rig.
- `filter_percentile` (default `2.5`): the worst percentage of observations removed per camera between the two optimization passes.

The returned `ExtrinsicCalibrationResult` carries the calibrated volume plus diagnostics:

```python
result.capture_volume              # the calibrated CaptureVolume
result.intrinsic_refinement_gated  # True if refinement was requested but gated off
result.depth_ratios                # per-camera near/far depth ratios
result.synthesized_cam_ids         # cameras whose intrinsics started from a blind guess
result.intrinsic_estimates         # initial vs recovered f, k1, k2 per camera
result.dropped_static_markers      # static markers excluded for moving (ArUco path)
```

Check `intrinsic_refinement_gated` when you expected refinement: a `True` here with poor results usually means the target was not moved toward and away from the cameras enough.

`calibrate_extrinsics` accepts a `progress` callback (`progress=lambda pct, msg: print(pct, msg)`); it runs silently by default.

### Manual pipeline control

The one-call pipeline is built from pieces you can also drive yourself:

```python
volume = CaptureVolume.bootstrap(ext_points, cameras, constraints=constraints)
volume = volume.optimize(strict=False)
volume = volume.filter_by_percentile_error(2.5)
volume = volume.optimize(strict=False)
```

`bootstrap` estimates initial camera positions from pairwise PnP and triangulates 3D points.
`optimize` runs bundle adjustment.
Both return new `CaptureVolume` instances (the originals are unchanged).
Reach for this when you want a nonstandard sequence, for example extra filter passes or intermediate inspection.
Otherwise prefer `calibrate_extrinsics`, which adds the robust loss, the depth-ratio gate, and the static-marker guard.

## Step 6: Align to object coordinates

```python
sync_idx = volume.unique_sync_indices[len(volume.unique_sync_indices) // 2]
volume = volume.align_to_object(sync_idx)
```

This applies a similarity transform (Umeyama algorithm) that aligns the world coordinate frame to the board's position at the chosen sync index.
Pick a frame where the board is at your desired origin.

## Step 7: Inspect results

```python
print_extrinsic_report(volume)
```

The report shows optimization status, reprojection error percentiles, per-camera breakdowns, and scale accuracy metrics (if origin was set).

## Step 8: Save

```python
volume.save("capture_volume")
```

This writes three files to the directory: `camera_array.toml`, `image_points.csv`, and `world_points.csv`.
To reload later:

```python
volume = CaptureVolume.load("capture_volume")
cameras = CameraArray.from_toml("capture_volume/camera_array.toml")
```

## Calibrating without intrinsics

The pipeline above assumed Step 3 produced calibrated intrinsics.
You can skip it.
When a camera in the array has no intrinsics, `calibrate_extrinsics` synthesizes a starting guess from the resolution and recovers focal length and leading distortion during bundle adjustment.
Read [Skipping Intrinsic Calibration](extrinsic_calibration.md#skipping-intrinsic-calibration) first; the prerequisites (a target swept through depth, no fisheye cameras) are the same from a script as from the GUI.

This example uses an [ArUco marker set](aruco_calibration_set.md), the usual companion to an extrinsic-only project:

```python
from caliscope.api import CameraArray, calibrate_extrinsics, extract_image_points_multicam
from caliscope.core.aruco_marker import ArucoMarkerSet
from caliscope.core.constraints import ConstraintSet
from caliscope.trackers.aruco_tracker import ArucoTracker

# Camera array straight from the extrinsic videos: no intrinsics anywhere
extrinsic_videos = {0: "extrinsic/cam_0.mp4", 1: "extrinsic/cam_1.mp4", 2: "extrinsic/cam_2.mp4"}
cameras = CameraArray.from_video_metadata(extrinsic_videos)

# Marker set defines the targets and their rigidity
marker_set = ArucoMarkerSet.from_toml("calibration/targets/aruco_marker_set.toml")
tracker = ArucoTracker(dictionary=marker_set.dictionary, marker_set=marker_set)
constraints = ConstraintSet.from_marker_set(marker_set)

ext_points = extract_image_points_multicam(extrinsic_videos, tracker)
result = calibrate_extrinsics(ext_points, cameras, constraints)
```

Leave `refine_intrinsics` at its default of `True` here.
The GUI forces this choice when cameras lack intrinsics; the API trusts you.
Passing `False` with uncalibrated cameras leaves the blind resolution-based guess in place, which produces a confidently wrong calibration.
Afterward, the diagnostics tell you how it went:

```python
if result.intrinsic_refinement_gated:
    print("Refinement gated off; the calibration rests on blind-guess intrinsics.")
    print(f"Per-camera depth ratios: {result.depth_ratios}")

for est in result.intrinsic_estimates:
    print(f"cam {est.cam_id}: f {est.f_initial:.0f} -> {est.f_recovered:.0f}")

result.capture_volume.save("capture_volume")
```

A gated result here deserves suspicion, not a save: with no prior intrinsics to fall back on, the gate means the recovered geometry is built on the synthesized guess.
Re-record with the marker moving toward and away from the cameras.

A fisheye camera (one with `fisheye = true` in its camera data) cannot take this path at all; `calibrate_extrinsics` raises a `CalibrationError` for it.
Calibrate fisheye cameras intrinsically first, then run the same extrinsic pipeline.
