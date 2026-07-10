# Extrinsic Calibration

Extrinsic calibration determines the position and orientation of every camera in a common 3D coordinate frame.
It requires synchronized video of a calibration target visible to multiple cameras.

## Calibration Targets

The GUI offers ChArUco boards and ArUco markers as extrinsic targets.
Chessboard targets also work through the [scripting API](scripting.md).
See [Calibration Targets](calibration_targets.md) for how to create and print them, and [The ArUco Calibration Set](aruco_calibration_set.md) for marker configuration.

## Workflow

### 1. Record and place files

Save synchronized videos to `calibration/extrinsic/` following the naming convention in [Project Setup](project_setup.md#stage-2-extrinsic-calibration).
Include a [`timestamps.csv`](project_setup.md#timestampscsv-format) if your cameras are not hardware-synchronized.

### 2. Extract

In the GUI, select the target type (ChArUco or ArUco) and run extraction.
Caliscope detects the target in every frame of every camera and saves the 2D corner locations to `image_points.csv`.

### 3. Calibrate

Click Calibrate. Caliscope estimates initial camera positions from pairwise PnP, then runs bundle adjustment to refine all cameras and 3D points jointly.

The target does not need to be visible in all cameras at once.
Caliscope chains pairwise relationships transitively: if cameras A–B and B–C each share a view of the target, A–C is inferred.
This supports surround-view setups where no single target position is visible to every camera.

The result is saved to `calibration/extrinsic/capture_volume/` as three files: `camera_array.toml`, `image_points.csv`, and `world_points.csv`.
An aniposelib-compatible export (`camera_array_aniposelib.toml`) is written to the workspace root for tools like [Pose2Sim](https://github.com/perfanalytics/pose2sim).

By default, bundle adjustment also refines each camera's focal length and leading distortion.
See [Extrinsic Calibration Reference](extrinsic_calibration_reference.md) for details on refinement and the depth-ratio gate.

### 4. Filter and re-optimize

The pipeline automatically removes the worst 2.5% of observations per camera and re-optimizes.
After this pass, a filter control lets you apply additional filtering and re-run bundle adjustment.
Per-camera filtering prevents cameras with fewer observations from being disproportionately stripped.

### 5. Set the origin

Choose a frame where the target sits at the desired world origin.
Caliscope applies a rigid transformation (rotation and translation) to align the coordinate frame to the board's position.
Setting the origin enables [scale accuracy metrics](extrinsic_calibration_reference.md#quality-metrics).

### 6. Rotate axes

90-degree rotation buttons align the coordinate axes with your lab conventions (Y-up vs Z-up, room orientation).

## Recording Tips

- The target does not need to be visible in all cameras at once, but each camera pair must share visibility at some point.
- Move the target toward and away from the cameras, not just across their views. Depth variation is what makes focal length observable.
- Use targets large enough to detect across your capture volume.
- Move slowly, use manual focus, and light the scene well.
- Place the target at the desired world origin in at least one frame.

## Scripting

Extrinsic calibration can be run from Python without the GUI.
See the [Scripting API](scripting.md#step-4-extract-time-aligned-points-for-extrinsic-calibration) for a walkthrough.
