# Triangulation API

Beyond the GUI interface, Caliscope provides a Python API for programmatically triangulating 2D points based on a calibrated camera array. This is intended for batch processing, custom workflows, or integration with other tools.

## Basic Usage

The core function for triangulation is `triangulate_from_files()`, which takes paths to your camera calibration and 2D point data, then returns triangulated 3D points.

```python
from pathlib import Path
from caliscope.triangulate.triangulation import triangulate_from_files

# Define paths
camera_array_path = Path("/path/to/project/calibration/extrinsic/capture_volume/camera_array.toml")
image_point_path = Path("/path/to/project/recordings/trial_1/POSE/xy_POSE.csv")
output_path = Path("/path/to/project/recordings/trial_1/POSE/xyz_POSE.csv")

# Perform triangulation
xyz_data = triangulate_from_files(
    camera_array_path=camera_array_path,
    image_point_path=image_point_path,
    output_path=output_path,
)

print(f"Triangulated {len(xyz_data)} points")
```

## Parameter Details

### Inputs

`camera_array_path` : Path to `camera_array.toml` containing camera calibration parameters. This file is produced during extrinsic calibration and saved in `calibration/extrinsic/capture_volume/`. It contains the intrinsic parameters (camera matrix, distortion coefficients) and extrinsic parameters (rotation, translation) for every camera.

`image_point_path` : Path to CSV file with 2D point data. The CSV must contain these columns:

- `sync_index`: Integer identifying the synchronized frame number across cameras
- `cam_id`: Integer identifying which camera captured the point
- `point_id`: Integer identifying which landmark/point is being tracked
- `img_loc_x`: Float X-coordinate in the image (in pixels)
- `img_loc_y`: Float Y-coordinate in the image (in pixels)
- `frame_time` (optional): Float timestamp in seconds. If present, it is propagated to the output DataFrame.

Each row represents a single 2D point observed by a specific camera at a specific time. Multiple cameras observing the same `point_id` at the same `sync_index` enables triangulation.

`output_path` (optional): Path where triangulated 3D points will be saved as a CSV file. If not provided, results are returned but not saved to disk. The function will create parent directories if they don't exist.

### Returned Value

Pandas DataFrame containing triangulated 3D points with columns:

- `sync_index`: Integer identifying the synchronized frame number
- `point_id`: Integer identifying which landmark/point was triangulated
- `x_coord`: Float X-coordinate in 3D space (meters)
- `y_coord`: Float Y-coordinate in 3D space (meters)
- `z_coord`: Float Z-coordinate in 3D space (meters)
- `frame_time`: Float timestamp in seconds (present when the input CSV includes `frame_time`)

Each row represents a single 3D point at a specific time. The coordinates are in the world coordinate system defined during extrinsic calibration (see [Extrinsic Calibration](extrinsic_calibration.md#origin-setting)).

### Notes

- For successful triangulation, each `point_id` must be visible in at least two cameras at the same `sync_index`.
- Points that cannot be triangulated (due to insufficient camera views) will not appear in the output DataFrame.
- The 3D coordinates are in meters, with scale determined by the physical size of the calibration target entered during extrinsic calibration.
