# Scripting API

Caliscope's calibration pipeline is available as Python functions. You can run intrinsic and extrinsic calibration from a script without the GUI.

The standard install includes the API:

```bash
uv pip install caliscope        # library only
uv pip install caliscope[gui]   # adds desktop app and 3D visualization
```

Everything below uses the `caliscope.api` module. A complete working script is in `scripts/demo_api.py`.

## Imports

```python
from pathlib import Path
from caliscope.api import (
    Charuco, CharucoTracker, CameraArray,
    CaptureVolume, extract_image_points, extract_image_points_multicam,
    calibrate_intrinsics,
)
from caliscope.reporting import (
    print_intrinsic_report, print_extrinsic_report, print_camera_pair_coverage,
)
```

`caliscope.api` re-exports the domain classes and calibration functions. `caliscope.reporting` provides Rich terminal output for quality inspection.

## Step 1: Define the calibration target

Create a Charuco board matching the physical board you recorded:

```python
charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=3.0)
tracker = CharucoTracker(charuco)
```

`square_size_cm` sets the scale of all calibrated coordinates. A 3.0 cm square produces corners spaced 0.03 m apart in object space. Measure your printed board and use the actual size.

## Step 2: Build a camera array from video metadata

```python
intrinsic_videos = {0: Path("intrinsic/cam_0.mp4"), 1: Path("intrinsic/cam_1.mp4")}
cameras = CameraArray.from_video_metadata(intrinsic_videos)
```

This reads resolution and frame rate from each video and creates uncalibrated `CameraData` entries. The dictionary keys are camera IDs, which must stay consistent across intrinsic and extrinsic videos.

## Step 3: Intrinsic calibration

For each camera, extract charuco corners from the intrinsic video, then solve for the camera matrix and distortion coefficients:

```python
for cam_id, video in intrinsic_videos.items():
    points = extract_image_points(video, cam_id, tracker, frame_step=5)
    output = calibrate_intrinsics(points, cameras[cam_id])
    cameras[cam_id] = output.camera

    print_intrinsic_report(output)
```

`frame_step=5` processes every 5th frame. Intrinsic calibration needs roughly 30 diverse frames, so skipping frames saves time without sacrificing quality.

`extract_image_points` shows a Rich progress bar by default. Pass `progress=None` to suppress it.

`calibrate_intrinsics` returns an `IntrinsicCalibrationOutput` containing the calibrated `CameraData` and an `IntrinsicCalibrationReport` with RMSE and coverage metrics. `print_intrinsic_report` prints these with color-coded quality badges.

## Step 4: Extract time-aligned points for extrinsic calibration

```python
extrinsic_videos = {0: Path("extrinsic/cam_0.mp4"), 1: Path("extrinsic/cam_1.mp4")}
ext_points = extract_image_points_multicam(extrinsic_videos, tracker)
```

This function reads all videos concurrently (one thread per camera), aligns frames by timestamp, and runs the tracker on each time-aligned moment. The result is a single `ImagePoints` DataFrame with observations from all cameras.

If your cameras were not hardware-synchronized, pass a timestamps file:

```python
ext_points = extract_image_points_multicam(extrinsic_videos, tracker, timestamps="timestamps.csv")
```

`frame_step` works on time-aligned moments, not raw frames. `frame_step=10` processes every 10th synchronized moment.

### Check camera pair coverage

Before calibrating, verify that camera pairs share enough observations:

```python
print_camera_pair_coverage(ext_points)
```

This prints a lower-triangle grid of shared observation counts and flags structural problems (e.g., a camera pair with zero overlap).

## Step 5: Bootstrap and optimize

```python
volume = CaptureVolume.bootstrap(ext_points, cameras)
volume = volume.optimize(strict=False)
```

`bootstrap` estimates initial camera positions from pairwise PnP, then triangulates 3D points. `optimize` runs bundle adjustment. Both return new `CaptureVolume` instances (the originals are unchanged).

### Filter outliers and re-optimize

```python
volume = volume.filter_by_percentile_error(2.5)
volume = volume.optimize(strict=False)
```

`filter_by_percentile_error(2.5)` removes the worst 2.5% of observations per camera, then re-triangulates. A second optimization pass on the cleaned data typically reduces reprojection error.

## Step 6: Align to object coordinates

```python
sync_idx = volume.unique_sync_indices[len(volume.unique_sync_indices) // 2]
volume = volume.align_to_object(sync_idx)
```

This applies a similarity transform (Umeyama algorithm) that aligns the world coordinate frame to the board's position at the chosen sync index. Pick a frame where the board is at your desired origin.

## Step 7: Inspect results

```python
print_extrinsic_report(volume)
```

The report shows optimization status, reprojection error percentiles, per-camera breakdowns, and scale accuracy metrics (if origin was set).

## Step 8: Save

```python
volume.save("capture_volume")
```

This writes three files to the directory: `camera_array.toml`, `image_points.csv`, and `world_points.csv`. To reload later:

```python
volume = CaptureVolume.load("capture_volume")
cameras = CameraArray.from_toml("capture_volume/camera_array.toml")
```

## Progress reporting

All extraction functions show Rich progress bars by default. Three ways to control this:

```python
# Default: Rich progress bar (auto-created, auto-cleaned-up)
points = extract_image_points(video, cam_id, tracker)

# Silent: no output
points = extract_image_points(video, cam_id, tracker, progress=None)

# Custom: implement the ProgressCallback protocol
points = extract_image_points(video, cam_id, tracker, progress=my_callback)
```

The `ProgressCallback` protocol requires four methods: `on_video_start`, `on_frame`, `on_video_complete`, and `on_info`. See `caliscope.reporting.RichProgressBar` for the reference implementation.
