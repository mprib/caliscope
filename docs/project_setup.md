# Workspace Setup

This document explains how to structure your workspace for multicamera calibration and motion capture with Caliscope.

## Initial Project Structure

When you create a new project, Caliscope automatically creates the necessary directory structure:

```
workspace/
├── calibration/
│   ├── targets/          # Auto-created calibration target configurations
│   ├── intrinsic/        # Per-camera calibration videos (unsynchronized)
│   └── extrinsic/        # Multi-camera calibration videos (synchronized)
└── recordings/           # Motion capture sessions
```

The application monitors these directories and automatically updates when files are added or changed.

## Camera Identification

Cameras are identified by integer IDs assigned through your video file naming. Video files must follow the naming convention `cam_N.mp4`, where N is the camera ID (e.g., `cam_0.mp4`, `cam_1.mp4`, `cam_2.mp4`).

- Camera IDs can be any non-negative integer
- Camera IDs do not need to be contiguous (e.g., `cam_0.mp4`, `cam_3.mp4`, `cam_7.mp4` is valid)
- The camera set is determined from the files present in the extrinsic calibration directory
- Camera IDs must remain consistent across intrinsic calibration, extrinsic calibration, and recording sessions

## Stage 1: Intrinsic Calibration

Intrinsic calibration determines each camera's internal properties (focal length, principal point, lens distortion).

Place one video per camera in `calibration/intrinsic/`. These videos **do not need to be synchronized**. Each video should show a calibration target (Charuco board, chessboard, or ArUco grid) being moved throughout the camera's field of view.

```
workspace/
└── calibration/
    └── intrinsic/
        ├── cam_0.mp4     # Individual camera recordings
        ├── cam_1.mp4     # No synchronization required
        └── cam_2.mp4
```

After calibration, each camera's intrinsic parameters are stored internally for use in extrinsic calibration.

## Stage 2: Extrinsic Calibration

Extrinsic calibration determines the spatial relationship between cameras (their positions and orientations in 3D space).

Place synchronized videos in `calibration/extrinsic/`. All cameras must observe the same physical space during the same time period.

```
workspace/
└── calibration/
    └── extrinsic/
        ├── cam_0.mp4
        ├── cam_1.mp4
        ├── cam_2.mp4
        └── timestamps.csv      # Optional: per-frame timing data
```

### Frame Synchronization

Caliscope needs to know which frames across cameras correspond to the same moment in time. There are two ways to provide this information:

**If you have per-frame timestamps**, place a `timestamps.csv` file in the recording directory. This is the most accurate option. It handles cameras with different frame rates, dropped frames, and different start times.

**If you don't have per-frame timestamps**, Caliscope infers timing from the video files themselves. This works when all cameras recorded at the same frame rate and captured a similar number of frames (e.g., videos that were trimmed to the same length in editing software). Caliscope saves its assumptions as `inferred_timestamps.csv` in the recording directory so you can inspect them.

### `timestamps.csv` Format

The file must have two columns: `cam_id` and `frame_time`.

```csv
cam_id,frame_time
0,927387.33536115
1,927387.50128975
2,927387.3530109001
0,927387.50643105
1,927387.51819965
2,927387.5063038999
0,927387.6684489499
1,927387.66848565
...
```

- **cam_id**: Must match the camera IDs from your video filenames
- **frame_time**: Numerical timestamp for when the frame was captured (e.g., from Python's `time.perf_counter()`)
- Rows can be in any order
- Cameras do not need the same number of frames or the same start time

Caliscope automatically aligns the videos during processing, inserting blank frames where necessary to maintain temporal correspondence.

### Calibration Output

After successful extrinsic calibration, Caliscope creates output in subdirectories:

```
workspace/
└── calibration/
    └── extrinsic/
        ├── cam_0.mp4
        ├── cam_1.mp4
        ├── cam_2.mp4
        ├── timestamps.csv           # If per-frame timing was provided
        ├── inferred_timestamps.csv  # Written by Caliscope when no timestamps.csv exists
        ├── CHARUCO/                 # Extraction output (tracker name varies)
        │   └── image_points.csv
        └── capture_volume/          # Calibration result
            ├── camera_array.toml
            ├── image_points.csv
            └── world_points.csv
```

The `capture_volume/` directory contains the complete calibrated camera system and can be used for 3D reconstruction of motion capture data.

## Stage 3: Recording and Reconstruction

For each motion capture session, create a subfolder within `recordings/` and populate it with synchronized videos. The same synchronization rules apply: provide a `timestamps.csv` if you have per-frame timing, otherwise Caliscope infers from video metadata.

```
workspace/
└── recordings/
    └── walking_trial/              # Name the folder descriptively
        ├── cam_0.mp4
        ├── cam_1.mp4
        ├── cam_2.mp4
        └── timestamps.csv          # Optional: same format as extrinsic
```

After processing with a motion tracking system (e.g., POSE, HAND, HOLISTIC), output files are created in a tracker-named subdirectory:

```
workspace/
└── recordings/
    └── walking_trial/
        ├── cam_0.mp4
        ├── cam_1.mp4
        ├── cam_2.mp4
        ├── timestamps.csv                  # If provided
        └── POSE/                           # Output subdirectory (tracker name)
            ├── camera_array.toml           # Snapshot of calibration used
            ├── xy_POSE.csv                 # 2D tracked points per camera
            ├── xyz_POSE.csv                # Triangulated 3D points
            ├── xyz_POSE_labelled.csv       # Wide-format 3D data with named columns
            └── xyz_POSE.trc                # OpenSim format for biomechanical analysis
```

## Output Files

### xy_[tracker].csv
2D landmark coordinates detected in each camera's view. Contains columns: `sync_index`, `cam_id`, `point_id`, `img_loc_x`, `img_loc_y`.

### xyz_[tracker].csv
Triangulated 3D coordinates in long format. Contains columns: `sync_index`, `point_id`, `x_coord`, `y_coord`, `z_coord`, plus metadata fields. Each row represents one landmark point at one time frame.

### xyz_[tracker]_labelled.csv
Wide-format 3D data with named columns (e.g., `nose_x`, `nose_y`, `nose_z`, `left_shoulder_x`, ...). Each row represents one time frame with all landmarks as separate columns. This format is easier for analysis in spreadsheet applications or data science tools like pandas.

### xyz_[tracker].trc
Track Row Column format for OpenSim and other biomechanical modeling software. Contains the same 3D trajectory data formatted according to OpenSim specifications, with landmark names and units (meters).

### camera_array.toml
A snapshot of the camera calibration (intrinsic and extrinsic parameters) used for this specific reconstruction. This ensures reproducibility even if the calibration is later updated.
