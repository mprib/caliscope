# Reconstruction: Landmark Triangulation from Motion Capture

The reconstruction pipeline transforms synchronized videos into 3D motion trajectories through two stages:

1. **2D landmark detection**: a tracker processes each camera's video to identify anatomical landmarks (e.g., joint positions) in every frame
2. **3D triangulation**: using the calibrated camera system, corresponding 2D observations from multiple cameras are triangulated into 3D world coordinates

The pipeline uses the camera intrinsics and extrinsics established during calibration to locate landmarks in physical space.

## Available Trackers

### ONNX Trackers

Caliscope uses ONNX pose estimation models for 2D landmark tracking. Several RTMPose Halpe26 models (tiny through xlarge, 26 body landmarks) ship as built-in model cards and can be downloaded in-app on first use. You can also load custom models exported from SLEAP, DeepLabCut, RTMPose, or other frameworks — see [Custom ONNX Trackers](onnx_trackers.md) for setup instructions.

The reconstruction pipeline is a convenience tool for verifying calibration quality and quick landmark export. For production reconstruction workflows, tools like [anipose](https://anipose.readthedocs.io/) and [Pose2Sim](https://github.com/perfanalytics/pose2sim) are better suited. Caliscope's aniposelib-compatible camera export makes it straightforward to calibrate here and hand off to those tools.

## Workflow

1. Navigate to the **Reconstruction** tab
2. Select the recording you want to process from the list
   - Recordings are detected automatically from subfolders within `recordings/` that contain synchronized videos
   - You may need to reload the workspace if recordings were added while the application was running
3. Choose a tracker from the dropdown menu
4. Click **Process** to begin landmark tracking and triangulation
5. Results appear in the 3D viewer when processing completes
6. Open the recording's output subfolder to access trajectory files

## Output Files

After processing, output is saved to a subfolder named after the tracker within the recording directory (e.g., `recordings/walking/POSE/`).

| File | Format | Description |
|------|--------|-------------|
| `xy_{TRACKER}.csv` | Long CSV | 2D tracked points per camera (sync_index, cam_id, point_id, img_loc_x, img_loc_y, frame_time) |
| `xyz_{TRACKER}.csv` | Long CSV | Triangulated 3D points (sync_index, point_id, x_coord, y_coord, z_coord, frame_time) |
| `xyz_{TRACKER}_labelled.csv` | Wide CSV | Named columns (e.g., nose_x, nose_y, nose_z, left_shoulder_x, ...) |
| `xyz_{TRACKER}.trc` | TRC | OpenSim-compatible format for biomechanical modeling |
| `camera_array.toml` | TOML | Snapshot of the camera calibration used for this reconstruction |

### Coordinate Units

All 3D coordinates are in **meters**. The physical scale is determined by the calibration target dimensions you entered during extrinsic calibration. See [Calibration Targets](calibration_targets.md#physical-size-and-world-scale) for details on how the scale chain propagates from board geometry to world coordinates.

## Per-Recording Camera Snapshot

Each reconstruction saves a copy of `camera_array.toml` alongside its output files. This ensures that recalibrating your camera system does not invalidate previous reconstruction results. Each recording retains the exact calibration parameters used to produce it.

In longitudinal studies where camera positions may shift between sessions, this prevents the need to reprocess archived recordings.

## Practical Recording Guidelines

### Minimize Motion Blur

Motion blur substantially compromises landmark recognition. To reduce blur:

- Use higher frame rates (e.g., 60 fps or above for dynamic movements)
- Increase lighting to maintain exposure at faster shutter speeds
- Avoid slow shutter speeds that allow excessive motion during exposure

### Lighting

- Ensure adequate, even lighting across the capture volume
- Avoid harsh shadows or high-contrast regions that can confuse trackers
- Diffuse lighting generally produces more consistent tracking than point sources
