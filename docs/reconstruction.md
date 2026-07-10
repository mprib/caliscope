# Reconstruction

Caliscope's reconstruction pipeline is a convenience tool for verifying calibration quality and quick landmark export.
For production 3D reconstruction, use [Pose2Sim](https://github.com/perfanalytics/pose2sim) or [anipose](https://anipose.readthedocs.io/).
Caliscope's aniposelib-compatible camera export lets you calibrate here and hand off to those tools.

The pipeline tracks 2D landmarks with ONNX pose models, then triangulates them into 3D using the calibrated cameras.
Several RTMPose Halpe26 models ship as built-in model cards and can be downloaded in-app.
Custom models from SLEAP, DeepLabCut, or other frameworks also work.
See [Custom ONNX Trackers](onnx_trackers.md) for setup.

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

After processing, output is saved to a subfolder named after the tracker within the recording directory (e.g., `recordings/walking/ONNX_rtmpose_t_halpe26/`).

| File | Format | Description |
|------|--------|-------------|
| `xy_{tracker}.csv` | Long CSV | 2D tracked points per camera (sync_index, cam_id, frame_index, frame_time, point_id, img_loc_x, img_loc_y, obj_loc_x, obj_loc_y) |
| `xyz_{tracker}.csv` | Long CSV | Triangulated 3D points (sync_index, point_id, x_coord, y_coord, z_coord, frame_time) |
| `xyz_{tracker}_labelled.csv` | Wide CSV | Named columns (e.g., nose_x, nose_y, nose_z, left_shoulder_x, ...) |
| `xyz_{tracker}.trc` | TRC | OpenSim-compatible format for biomechanical modeling |
| `camera_array.toml` | TOML | Snapshot of the camera calibration used for this reconstruction |

### Coordinate Units

All 3D coordinates are in **meters**. The physical scale is determined by the calibration target dimensions you entered during extrinsic calibration. See [Calibration Targets](calibration_targets.md#physical-size-and-world-scale) for how the board measurement you entered becomes the unit of all 3D output.

## Per-Recording Camera Snapshot

Each reconstruction saves a copy of `camera_array.toml` alongside its output files, so recalibrating your camera system does not invalidate previous results. Each recording retains the exact calibration parameters used to produce it.

In longitudinal studies where camera positions may shift between sessions, this prevents the need to reprocess archived recordings.

## Recording Tips

Move slowly, light the scene evenly, and avoid harsh shadows.
Higher frame rates reduce motion blur but need more light.
