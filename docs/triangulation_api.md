# Triangulation API

Beyond the GUI interface, Caliscope provides a Python API for programmatically triangulating 2D points based on a calibrated camera array. This is intended for batch processing, custom workflows, or integration with other tools.

## Basic Usage

The core function for triangulation is `triangulate_from_files()`, which takes paths to your configuration file and 2D point data, then returns triangulated 3D points. With an optional argument these can be saved to a `csv`.

```python
from pathlib import Path
from caliscope.triangulate.triangulation import triangulate_from_files

# Define paths
config_path = Path("/path/to/project/config.toml")
xy_path = Path("/path/to/project/xy_data.csv")
output_path = Path("/path/to/project/xyz_data.csv")

# Perform triangulation
xyz_data = triangulate_from_files(
    config_path=config_path,
    xy_path=xy_path, 
    output_path=output_path
)

print(f"Triangulated {len(xyz_data)} points")
```

## Parameter Details

### Inputs  

`config_path` : Path to config.toml containing camera calibration parameters. This file should be located in the root of the project workspace and contains essential camera parameters like intrinsics, extrinsics, and distortion coefficients.

`xy_path` : Path to CSV file with 2D point data. The CSV must contain these columns:

- sync_index: Integer identifying the synchronized frame number across cameras
- port: String or integer identifying which camera captured the point
- point_id: Integer identifying which landmark/point is being tracked
- img_loc_x: Float X-coordinate in the image (in pixels)
- img_loc_y: Float Y-coordinate in the image (in pixels)

Each row represents a single 2D point observed by a specific camera at a specific time. Multiple cameras observing the same point_id at the same sync_index enables triangulation.

`output_path`(optional): Path where triangulated 3D points will be saved as a CSV file.  If not provided, results are returned but not saved to disk. The function will create parent directories if they don't exist.

### Returned Value 

DataFrame containing triangulated 3D points with columns:

- sync_index: Integer identifying the synchronized frame number
- point_id: Integer identifying which landmark/point was triangulated
- x_coord: Float X-coordinate in 3D space 
- y_coord: Float Y-coordinate in 3D space 
- z_coord: Float Z-coordinate in 3D space 

Each row represents a single 3D point at a specific time.  The coordinates are in the world coordinate system defined during the calibration process.

### Notes

- For successful triangulation, each point_id must be visible in at least
  two cameras at the same sync_index.
- Points that cannot be triangulated (due to insufficient camera views)
  will not appear in the output DataFrame.
