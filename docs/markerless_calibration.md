# Markerless Calibration (Experimental)

This is an experimental scripting alternative to [target-based calibration](calibration_targets.md).
The standard workflow uses a board or marker with identifiable corners and known dimensions.
Here, a pose tracker identifies matching body keypoints across synchronized camera views to estimate the cameras' relative positions and orientations.

Without a target of known size, the result needs a separate scale measurement and coordinate frame.
The example uses a measured distance between cameras for scale, Caliscope's built-in vertical estimate to find which direction is up, and a low tracked point to set the floor height.
The vertical estimate runs a [GeoCalib](https://github.com/cvg/GeoCalib) model converted to ONNX, so it needs nothing beyond the `[tracking]` extra.
The example then centers the origin between the cameras.

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

Record one calibration subject walking through the shared capture space.
Camera views must overlap through enough of the motion to connect the rig.
Other people can be visible, but the intended subject's bounding boxes must cover the most image area in each camera, counting each pixel once.

The example first detects people and links their boxes into camera-local tracks.
It selects the primary track by the image area its boxes cover, excludes boxes near the image edge, and estimates keypoints only for the retained observations.
Track IDs are local to each camera; this workflow relies on the recording protocol to select the same person across views.
It does not match several people across cameras or join fragmented tracks automatically.

The box-edge margin is an explicit threshold in original image pixels.
It is separate from detector confidence and overlap-suppression settings.
The generic `extract_image_points_multicam` entry point remains available; with a detector-backed ONNX tracker, it chooses the largest box independently in each frame.

Each video needs the same camera ID it had during intrinsic calibration.
Choose the alignment policy by building a `SynchronizedTimestamps` object and passing it to `track_people`.
`SynchronizedTimestamps.from_anchor_frames` takes one frame per camera known to show the same instant, and assumes each camera has a constant frame interval.
`SynchronizedTimestamps.from_csv_path` uses per-frame timestamps, and `SynchronizedTimestamps.from_video_paths` assumes the videos start and stop together.
The tracking workflow does not infer timing offsets from body motion.
The generic [multicamera extraction workflow](scripting.md#step-4-extract-time-aligned-points-for-extrinsic-calibration) still supports timestamp-based synchronization.

Measure the distance between two cameras in meters.
This measurement supplies the metric scale that pose keypoints cannot supply on their own.

For sideways videos, pass `rotation_counts` to `track_people`, for example `rotation_counts={2: 1}` to turn camera 2 by one quarter turn for tracking.
Saved boxes record the rotation for the keypoint pass; both boxes and keypoints return in the original image coordinates.

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
    estimate_keypoints,
    estimate_vertical,
    track_people,
)
from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps
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
EDGE_MARGIN_PX = 24  # Source-image pixels; set for your recording.
BASELINE = CameraDistance(cam_a=0, cam_b=2, meters=3.20)
OUTPUT_DIR = Path("markerless_capture_volume")

# Load the packaged RTMPose-l Halpe26 card without starting the GUI.
card_path = Path(str(files("caliscope.trackers.model_cards"))) / "rtmpose_l_halpe26.toml"
card = ModelCard.from_toml(card_path, models_dir=MODELS_DIR)
if not card.onnx_exists:
    download_and_extract_model(card, MODELS_DIR)
if not card.detector.onnx_exists:
    download_and_extract_model(card.detector, MODELS_DIR)
tracker = OnnxTracker(card)

# Detect and track boxes, then select the most prominent person per camera.
cameras = CameraArray.from_toml(CAMERA_ARRAY_TOML)
timestamps = SynchronizedTimestamps.from_anchor_frames(VIDEOS, {cam_id: 0 for cam_id in VIDEOS})
box_tracks = track_people(VIDEOS, tracker.detector, timestamps=timestamps, frame_step=1)
subject_ids = box_tracks.primary_track_ids()
subject_boxes = box_tracks.filter_to_tracks(subject_ids)
usable_boxes = subject_boxes.exclude_edge_boxes(margin_px=EDGE_MARGIN_PX)

# Estimate keypoints only in the retained boxes.
image_points = estimate_keypoints(
    VIDEOS, tracker, usable_boxes, min_confidence=card.confidence_threshold
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

The first run downloads the RTMPose and detector weights if they are absent from `MODELS_DIR`.
The detector sees every processed frame; keypoint inference sees only the retained subject boxes.
Each pass reads the source videos; no wall-time speedup is assumed.
The GeoCalib model also downloads to `MODELS_DIR` the first time `estimate_vertical` runs.

`BoxTracks` supports `to_csv` and `from_csv`, so you can save detections and change the selected track or edge margin without repeating detection.
Keypoints already saved as `ImagePoints` can also be reloaded.
Selecting previously unprocessed boxes requires keypoint inference for those boxes.
The returned image points use `object_id=0` for the selected person and retain camera-local `track_id` and keypoint confidence as additional columns.
Missing observations remain absent; no boxes or keypoints are interpolated by these stages.

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

## Calibration and motion capture

The person detector and pose estimator supply image observations for calibration and reconstruction.
Calibration uses those correspondences to estimate camera poses; reconstruction uses known camera poses to triangulate them.
Both depend on selecting the same person across views.

Caliscope's reconstruction supports calibration inspection and landmark export.
For downstream motion-capture processing, export the camera calibration to tools such as Pose2Sim or anipose.
See [the reconstruction scope](reconstruction.md) and [aniposelib export](scripting.md#aniposelib-export).
