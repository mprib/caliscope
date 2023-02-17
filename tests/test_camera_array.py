# I recognize this is not really structured as a test, I just need
# to mock something up and don't want to destroy my old F5 test.
# This may be the beginning of me moving into something that is
# a little more "TDD" like. Not full unit tests, but something more
# stable and open to automation.
#%%
from pathlib import Path

from calicam.cameras.camera_array import CameraArray
from calicam.cameras.camera_array_builder import CameraArrayBuilder
from calicam.calibration.bundle_adjustment.bundle_adjustment_data import (
    BundleAdustmentData,
    get_bundle_adjustment_data,
)


from calicam import __root__

session_directory = Path(__root__, "tests", "5_cameras")
config_path = Path(session_directory, "config.toml")
array_builder = CameraArrayBuilder(config_path)
camera_array = array_builder.get_camera_array()

# session_directory = Path(repo, "sessions", "iterative_adjustment")
points_csv_path = Path(session_directory, "recording", "triangulated_points.csv")

bund_adj_data = get_bundle_adjustment_data(points_csv_path)
print(f"Optimizing initial camera array configuration ")
# camera_array.optimize(point_data, output_path = points_csv_path.parent)
camera_array.bundle_adjust(bund_adj_data, output_path=points_csv_path.parent)

# %%
