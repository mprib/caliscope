# Markerless Calibration (Experimental)

This is an experimental scripting alternative to [target-based calibration](calibration_targets.md).
The standard workflow uses a board or marker with identifiable corners and known dimensions.
Here, a pose tracker identifies matching body keypoints across synchronized camera views to estimate the cameras' relative positions and orientations.

Without a target of known size, the result needs a separate scale measurement and coordinate frame.
The example uses a measured distance between cameras for scale, GeoCalib to estimate which direction is up, and a low tracked point to set the floor height.
It then centers the origin among the cameras.

!!! warning "Experimental"
    This workflow has been exercised in synthetic tests and limited real-world demonstrations.
    Its accuracy and reliability across recording conditions have not been established.

## Install

The workflow runs from Python and requires pose tracking:

```bash
uv pip install "caliscope[tracking]"
```

## Prepare the recording

You need calibrated intrinsics for every camera.
Follow the [intrinsic-calibration scripting workflow](scripting.md#step-3-intrinsic-calibration) to create the camera-array TOML.

Record a single person while each camera sees the person at the same time for enough of the motion to connect the rig.
Each video needs the same camera ID it had during intrinsic calibration.
Use a `timestamps.csv` file when the recordings did not start at exactly the same time.
Its `frame_time` values must use a common time basis across cameras.
The pipeline uses those supplied timestamps to form synchronized moments.
It does not infer timing offsets from body motion.
See [the timestamp CSV schema](scripting.md#step-4-extract-time-aligned-points-for-extrinsic-calibration).

Measure the distance between two cameras in meters.
This measurement supplies the metric scale that pose keypoints cannot supply on their own.

For sideways videos, pass `rotation_counts` to `extract_image_points_multicam`, for example `rotation_counts={2: 1}` to turn camera 2 by one quarter turn for tracking.
The rotation is applied only while tracking, and the extracted points return in the original image coordinates.

## Run the calibration

Edit the paths, camera IDs, and measured distance in this script.
The camera-array TOML must contain the intrinsic matrices and distortion coefficients from your intrinsic calibration.

```python
from importlib.resources import files
from pathlib import Path

from caliscope import MODELS_DIR
from caliscope.api import (
    CameraArray,
    CameraDistance,
    calibrate_extrinsics,
    estimate_vertical,
    extract_image_points_multicam,
)
from caliscope.reporting import print_extrinsic_report
from caliscope.trackers.model_card import ModelCard
from caliscope.trackers.model_download import download_and_extract_model
from caliscope.trackers.onnx_tracker import OnnxTracker

# Change these paths and camera IDs to match your intrinsic calibration and videos.
CAMERA_ARRAY_TOML = Path("intrinsics/camera_array.toml")
VIDEOS = {
    0: Path("markerless/cam_0.mp4"),
    1: Path("markerless/cam_1.mp4"),
    2: Path("markerless/cam_2.mp4"),
}
TIMESTAMPS = None  # Or: Path("markerless/timestamps.csv")
BASELINE = CameraDistance(cam_a=0, cam_b=2, meters=3.20)
OUTPUT_DIR = Path("markerless_capture_volume")

# Load the packaged RTMPose-l Halpe26 card without starting the GUI.
card_path = Path(str(files("caliscope.trackers.model_cards"))) / "rtmpose_l_halpe26.toml"
card = ModelCard.from_toml(card_path, models_dir=MODELS_DIR)
if not card.onnx_exists:
    download_and_extract_model(card, MODELS_DIR)
tracker = OnnxTracker(card)

# Load real intrinsics and track every synchronized moment.
cameras = CameraArray.from_toml(CAMERA_ARRAY_TOML)
image_points = extract_image_points_multicam(
    VIDEOS,
    tracker,
    frame_step=1,
    timestamps=TIMESTAMPS,
)

# Solve poses from unconstrained body keypoints.
# Keep the measured intrinsics fixed.
run = calibrate_extrinsics(
    image_points,
    cameras,
    constraints=None,
    refine_intrinsics=False,
)
volume = run.capture_volume

# Set +Z from GeoCalib's vertical estimate, then anchor scale and origin.
vertical = estimate_vertical(VIDEOS, cameras)
volume = volume.oriented(up=vertical.up_per_cam)
volume = volume.scaled(BASELINE)
volume = volume.grounded()
volume = volume.centered()

print_extrinsic_report(volume)
volume.save(OUTPUT_DIR)
volume.camera_array.to_aniposelib_toml(OUTPUT_DIR.parent / "camera_array_aniposelib.toml")
```

The first run downloads the RTMPose weights if they are absent from `MODELS_DIR`.
GeoCalib also downloads its field-net weights on its first use.

Grounding places the lowest tracked point at `Z=0`.
It does not detect the physical floor.
Use `lowest_point_height_m` when a reliably tracked low point, such as a foot keypoint, has a known height above the floor.
Use `mode="pooled_1st_percentile"` only when a few reconstructed points lie below the true floor.
See [the grounding options](scripting.md#aligning-to-the-floor-without-a-board) for examples.

## Inspect the calibration

```python
# Optional: view the calibrated cameras and tracked person in 3D.
# Requires: uv pip install "caliscope[gui]" (includes tracking).
from caliscope.gui import view_capture_volume

view_capture_volume(volume, wireframe=tracker.wireframe, fps=30)
```

Run the viewer from a standalone Python script.
It opens an interactive window and pauses the script until you close it.

The saved directory contains the camera array and the 2D and 3D points used for the result.
You can reopen it later without loading a tracker or model card.

```python
from caliscope.api import CaptureVolume
from caliscope.gui import view_capture_volume

volume = CaptureVolume.load("markerless_capture_volume")
view_capture_volume(volume, fps=30)
```

The viewer can display the points without a wireframe.
Pass the `wireframe` from a loaded model card when you want the Halpe26 skeleton overlay.
Its `fps` value sets the playback cadence for the available samples.
It does not reproduce the recording timestamps.

## What to check

Inspect the report, camera layout, and motion before using the calibration for measurement.
A low reprojection error and a plausible 3D view do not establish measurement accuracy.
Look for views that overlap through the motion, a vertical direction that agrees with the scene, and a scale that agrees with an independent measurement.
