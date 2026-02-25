# Capture Volume: Extrinsic Calibration

Extrinsic calibration determines the position and orientation of every camera in a common 3D coordinate frame. This process requires synchronized video of a calibration target visible to multiple cameras. Once complete, the calibrated camera array enables 3D triangulation of tracked landmarks from multiple camera views.

Caliscope uses a two-stage approach: first, it estimates relative camera positions from pairs of cameras that share a view of the calibration target; second, bundle adjustment simultaneously refines all camera positions and 3D point estimates to minimize reprojection error.

## Supported Calibration Targets

Caliscope supports two types of calibration targets for extrinsic calibration:

- **ChArUco Board**: A chessboard pattern with ArUco markers embedded in the black squares. Provides more detection points per frame (all interior corners) and is robust to partial occlusion.
- **ArUco Marker**: A single planar marker. Simpler to produce but provides only four corner points per detection.

See [Calibration Targets](calibration_targets.md) for detailed information on each target type, including how to generate and print them.

## Workflow

### 1. Recording and File Setup

Save synchronized videos to `project_root/calibration/extrinsic/` according to the naming convention outlined in [Project Setup](project_setup.md#stage-2-extrinsic-calibration). Ensure videos were synchronized during recording, or provide a [`timestamps.csv`](project_setup.md#timestampscsv) file for post-hoc synchronization.

### 2. Extraction

The first processing step is extraction. Select which calibration target type to use (ChArUco or ArUco), then run extraction. Caliscope detects the target in each frame of each camera and records the 2D image locations of detected corners or marker corners.

The extraction output is saved as `image_points.csv` in either `calibration/extrinsic/CHARUCO/` or `calibration/extrinsic/ARUCO/`, depending on the target type selected.

### 3. Initial Camera Positions from Pairwise Estimation

After extraction, Caliscope estimates the initial position and orientation of every camera. The calibration target does **not** need to be visible in all cameras simultaneously. Instead:

- For each pair of cameras that both see the target in the same frame, Caliscope estimates their relative position and orientation
- It chains these pairwise relationships together: if the relationship between cameras A and B is known, and the relationship between B and C is known, then A to C can be derived transitively
- This process repeats until all cameras that share any chain of connections have estimated positions

This means you can calibrate surround-view setups where no single position of the target is visible to all cameras at once. The pairwise estimates serve as the starting point for bundle adjustment.

### 4. Bundle Adjustment

Bundle adjustment is a nonlinear least-squares optimization that simultaneously refines all camera positions and all 3D point estimates to minimize reprojection error. Reprojection error is the distance (in pixels) between where a 3D point projects into a camera image using the current camera parameters, versus where the point was actually observed in that image.

Caliscope uses scipy's optimization routines to solve this large-scale problem, producing the final calibrated camera array. The optimized result is automatically saved to `calibration/extrinsic/capture_volume/` as three files:

- `camera_array.toml`: Camera intrinsics and extrinsics
- `image_points.csv`: The 2D observations used (potentially filtered)
- `world_points.csv`: The triangulated 3D points

These three files together form a complete snapshot of the calibrated capture volume.

An aniposelib-compatible version of the camera parameters is also written as `camera_array_aniposelib.toml` in the same directory. Tools that consume the aniposelib calibration format, such as [Pose2Sim](https://github.com/perfanalytics/pose2sim), can use this file directly.

### 5. Filter and Re-optimize

The initial calibration pipeline includes a built-in outlier removal pass that removes the worst 2.5% of observations per camera and re-optimizes. After this automatic pass completes, a filter control allows you to apply additional filtering. The default filter threshold also targets the 2.5th percentile, applied per camera (not globally). Per-camera filtering ensures that cameras with fewer observations are not disproportionately stripped. Removing additional outliers and re-running bundle adjustment can further improve the calibration result.

During re-optimization, the calibration controls remain visible but are disabled. Once re-optimization completes, the controls re-enable and the updated results are displayed.

### 6. Origin Setting

After calibration, you can choose a frame where the calibration target is positioned at the desired world origin. Caliscope applies a similarity transformation (using the Umeyama algorithm) that includes rotation, translation, and scale refinement to align the coordinate frame with the board's position in the selected frame.

**Setting the origin enables volumetric scale accuracy metrics** (see Quality Metrics below). It also establishes the world coordinate frame in a physically meaningful location, such as the floor of your capture volume.

### 7. Rotation Controls

After setting the origin, 90-degree rotation buttons allow you to align the coordinate axes with your lab conventions. For example, you may want Y-up versus Z-up, or you may want to rotate the axes to match the layout of your room or experimental apparatus.

### 8. Quality Metrics

Once the origin is set, Caliscope computes volumetric scale accuracy metrics:

**Pooled RMSE (in mm)**: This is the root-mean-square error of pairwise distances between reconstructed board corners, compared to the known board geometry. It measures how accurately the 3D reconstruction preserves physical scale.

Interpretation guidelines:
- **Good**: < 2mm
- **Acceptable**: 2-5mm
- **Concerning**: > 5mm

**Sparkline**: A small inline chart showing per-frame scale accuracy across the entire calibration sequence. This allows you to quickly identify frames with particularly high or low accuracy.

**Bias interpretation**: The scale error can reveal systematic biases:
- If errors are consistently positive (reconstructed distances larger than known distances), the entered board size may be slightly too small
- If errors are consistently negative (reconstructed distances smaller than known distances), the entered board size may be too large

This helps catch measurement errors in the board dimensions you entered during setup.

## Practical Recording Guidelines

### Coverage and Overlap
- Move the calibration target throughout the entire volume where the cameras' fields of view overlap
- Ensure there is sufficient overlap in the fields of view between different cameras
- With pairwise bootstrapping, the target does NOT need to be visible in all cameras at once, but each camera pair must share visibility at some point during the recording

### Board Size
- Use a board with sufficiently large markers for your capture volume
- Larger ArUco markers can be identified from farther away, enabling calibration of larger capture volumes
- For close-range setups, smaller markers provide higher precision

### Motion Blur
- Motion blur can substantially compromise corner detection
- Move the board slowly and smoothly
- Using a higher frame rate can reduce motion blur, but will require more light to maintain good illumination

### Focus
- Use manual focus if available to keep focus consistent throughout the recording
- Auto-focus can introduce inconsistencies as the board moves through the scene

### Origin Frame
- Setting the board origin is for convenience and not a strict requirement
- To use the board for origin setting, place it at the desired world origin position (e.g., touching the floor) in at least one frame
- Ensure the board is fully visible in that frame for best results
