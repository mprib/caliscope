# Markerless Calibration (Experimental)

Use this workflow when synchronized cameras recorded one moving person with overlapping views, but no calibration target.
It estimates the camera poses from pose keypoints, then anchors the result with an estimated vertical direction and a measured distance between two cameras.

You need a calibrated intrinsic matrix and distortion coefficients for every camera before you begin.
Follow the [intrinsic-calibration scripting workflow](scripting.md#step-3-intrinsic-calibration) to create the camera-array TOML.
The public markerless path rejects cameras that lack them.

!!! warning "Experimental"
    The calibration pipeline has synthetic coverage and limited real-world demonstrations.
    Broader validation is still pending.
    Treat a low reprojection error and a plausible 3D view as diagnostics, not proof that measurements are accurate.

## Install

The headless pipeline needs the tracking extra.

```bash
uv pip install "caliscope[tracking]"
```

The interactive viewer needs the GUI extra, which includes tracking.

```bash
uv pip install "caliscope[gui]"
```

Run the full recipe below as a standalone Python script on a desktop system.
The viewer opens a window and pauses the script until you close it.
It cannot run inside an existing Qt application.

For a headless install, omit `from caliscope.gui import view_capture_volume` and the final `view_capture_volume(...)` call.

## Prepare the recording

Record a single person while each camera sees the person at the same time for enough of the motion to connect the rig.
Each video needs the same camera ID it had during intrinsic calibration.
Use a `timestamps.csv` file when the recordings did not start at exactly the same time.
Its `frame_time` values must use a common time basis across cameras.
The pipeline uses those supplied timestamps to form synchronized moments.
It does not infer timing offsets from body motion.
See [the timestamp CSV schema](scripting.md#step-4-extract-time-aligned-points-for-extrinsic-calibration).

Measure the distance between two cameras in meters.
This measurement supplies the metric scale that pose keypoints cannot supply on their own.

If a video is sideways or upside down, set its `rotation_counts` entry to the number of quarter turns that makes the person upright for RTMPose.
For example, `1` is one quarter turn and `-1` is one quarter turn in the opposite direction.
The rotation is applied only while tracking, and the extracted points return in the original image coordinates.

## Run the calibration

Edit the paths, camera IDs, measured baseline, and playback frame rate in this script.
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
from caliscope.gui import view_capture_volume
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
ROTATION_COUNTS = {0: 0, 1: 0, 2: 0}
BASELINE = CameraDistance(cam_a=0, cam_b=2, meters=3.20)
FPS = 30
OUTPUT_DIR = Path("markerless_capture_volume")
ANIPOSELIB_TOML = OUTPUT_DIR.parent / "camera_array_aniposelib.toml"

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
    rotation_counts=ROTATION_COUNTS,
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
volume.camera_array.to_aniposelib_toml(ANIPOSELIB_TOML)

view_capture_volume(volume, wireframe=tracker.wireframe, fps=FPS)
```

`frame_step=1` processes every synchronized moment.
This gives pose calibration the most observations available from the recording.
Increase it only when runtime is a constraint and you still retain overlapping motion across the rig.

The first run downloads the RTMPose weights if they are absent from `MODELS_DIR`.
GeoCalib also downloads its field-net weights on its first use.

The script estimates vertical direction with GeoCalib, then makes that direction `+Z`.
It applies the measured `CameraDistance` after orientation, shifts the selected lowest tracked point to `Z=0`, and centers the camera layout in XY.
Grounding does not detect a physical floor.
Use `lowest_point_height_m` when a reliably tracked low point, such as a foot keypoint, has a known height above the floor.
Use `mode="pooled_1st_percentile"` only when a few reconstructed points lie below the true floor.
See [the grounding options](scripting.md#aligning-to-the-floor-without-a-board) for examples.

## Reload and inspect a saved calibration

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
Look for views that overlap through the motion, a vertical direction that agrees with the scene, and a scale that agrees with an independent measurement.

This workflow supplies metric scale only through the measured camera baseline.
The estimated vertical and the low-point grounding step set useful coordinates, but neither validates measurement accuracy.
