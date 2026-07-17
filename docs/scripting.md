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
    Charuco, CharucoTracker, CameraArray, CaptureVolume, ConstraintSet,
    extract_image_points, extract_image_points_multicam,
    calibrate_intrinsics, calibrate_extrinsics,
)
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

For a two-sided board on a real substrate, pass the measured `thickness_cm` so back-face detections are modeled at their true depth. See [Two-Sided Boards and Thickness](calibration_targets.md#two-sided-boards-and-thickness) for mounting and coverage requirements:

```python
charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=3.0, thickness_cm=0.6)
```

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
The extrinsic pipeline can recover intrinsics during bundle adjustment; see [Calibrating without intrinsics](#calibrating-without-intrinsics-experimental) below. This path is experimental.

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
run = calibrate_extrinsics(ext_points, cameras, constraints)
volume = run.capture_volume
```

The constraints feed the board's known corner geometry into bundle adjustment as rigidity information, which stabilizes the solution and locks world scale.
Build them from the same board definition you used for tracking.

Two keyword arguments matter most:

- `refine_intrinsics` (default `True`): re-estimate each camera's focal length and leading distortion jointly with the poses.
  Refinement is subject to the [depth-ratio gate](extrinsic_calibration_reference.md#the-depth-ratio-gate): if any camera saw the target over too narrow a depth range, refinement is disabled for the whole rig.
- `filter_percentile` (default `2.5`): the worst percentage of observations removed per camera between the two optimization passes.

The returned `CalibrationRun` carries the calibrated volume plus diagnostics:

```python
run.capture_volume              # the calibrated CaptureVolume
run.intrinsic_refinement_gated  # True if refinement was requested but gated off
run.synthesized_cam_ids         # cameras whose intrinsics started from a blind guess
run.intrinsic_estimates         # initial vs recovered f, k1, k2 per camera
run.dropped_static_markers      # static markers excluded for moving (ArUco path)
```

Per-camera near/far depth ratios are available via `compute_depth_ratios(run.capture_volume)` (`from caliscope.core.scale_accuracy import compute_depth_ratios`).

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

## Calibrating without intrinsics (experimental)

!!! warning "Experimental"
    This path passes synthetic tests but has not been validated on real-world data. The recommended workflow is to calibrate intrinsics first.

When a camera in the array has no intrinsics, `calibrate_extrinsics` synthesizes a starting guess from the resolution and recovers focal length and leading distortion during bundle adjustment.
See [Skipping Intrinsic Calibration](extrinsic_calibration_reference.md#skipping-intrinsic-calibration) for prerequisites.

```python
from caliscope.api import (
    ArucoMarkerSet, ArucoTracker, CameraArray, ConstraintSet,
    calibrate_extrinsics, extract_image_points_multicam,
)

cameras = CameraArray.from_video_metadata(extrinsic_videos)
marker_set = ArucoMarkerSet.from_toml("calibration/targets/aruco_marker_set.toml")
tracker = ArucoTracker(dictionary=marker_set.dictionary, marker_set=marker_set)
constraints = ConstraintSet.from_marker_set(marker_set)

ext_points = extract_image_points_multicam(extrinsic_videos, tracker)
result = calibrate_extrinsics(ext_points, cameras, constraints)
```
Calibrate fisheye cameras intrinsically first, then run the same extrinsic pipeline.

## Chessboard extrinsics

A plain chessboard can drive the same extrinsic pipeline.
The GUI does not offer it.
The board must meet the symmetry condition in the warning below.

```python
from caliscope.api import (
    Chessboard, ChessboardTracker, ConstraintSet,
    calibrate_extrinsics, extract_image_points_multicam,
)

chessboard = Chessboard(rows=6, columns=9, square_size_cm=3.0)
tracker = ChessboardTracker(chessboard)
constraints = ConstraintSet.from_chessboard(chessboard)

ext_points = extract_image_points_multicam(extrinsic_videos, tracker)
run = calibrate_extrinsics(ext_points, cameras, constraints)
```

`rows` and `columns` count internal corners, not squares.
`square_size_cm` is required here.
`ConstraintSet.from_chessboard` raises without it, because the recovered world scale would otherwise be wrong.
Detection is all-or-nothing, so a frame where any corner is cut off or covered contributes nothing.

!!! warning "Corner ordering and board symmetry"
    Use a board with one odd and one even inner-corner count, such as the example above.
    Current OpenCV releases resolve its orientation from the square coloring, and a regression test guards that behavior.
    A board with both counts even, or both odd, looks identical after a half turn, so no detector can tell the two orientations apart.
    The failure appears when two cameras see the board roughly a half turn apart, say one camera rolled 180 degrees, or two cameras looking down at a flat board from opposite ends of a room.
    The corner ids then reverse between the views, and triangulation silently pairs mismatched corners.
    With a symmetric board, every camera must see the board in a consistent orientation, and a ChArUco or ArUco target is the safer choice.
