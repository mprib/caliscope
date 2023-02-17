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

from calicam.cameras.synchronizer import Synchronizer
from calicam.triangulate.paired_point_stream import PairedPointStream   
from calicam.calibration.charuco import Charuco
from calicam.recording.recorded_stream import RecordedStream, RecordedStreamPool
from calicam.triangulate.array_triangulator import ArrayTriangulator

from calicam import __root__

session_directory = Path(__root__, "tests", "5_cameras")
config_path = Path(session_directory, "config.toml")

recording_directory = Path(session_directory, "recording")

points_csv_path = Path(recording_directory, "triangulated_points.csv")
bund_adj_data = get_bundle_adjustment_data(points_csv_path)

array_builder = CameraArrayBuilder(config_path)
camera_array = array_builder.get_camera_array()

print(f"Optimizing initial camera array configuration ")

# camera_array.optimize(point_data, output_path = points_csv_path.parent)
camera_array.bundle_adjust(bund_adj_data, output_path=points_csv_path.parent)
camera_array.update_extrinsic_params()
# %%

# Build streams from pre-recorded video
ports = [0, 1, 2, 3, 4]
charuco = Charuco(
    4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
)
recorded_stream_pool = RecordedStreamPool(
    ports, recording_directory, charuco=charuco
)

# synchronize videos
syncr = Synchronizer(recorded_stream_pool.streams, fps_target=100)
recorded_stream_pool.play_videos()

# create a commmon point finder to grab charuco corners shared between the pair of ports
point_stream = PairedPointStream(synchronizer=syncr)

# Build triangulator
# Note that this will automatically create the summarized output of the projected points
# this is just a temporary setup while I try to figure out something more suitable long-term

output_file = Path(recording_directory, "triangulated_points_post_BA.csv")
array_triangulator = ArrayTriangulator(camera_array, point_stream, output_file)
# %%
