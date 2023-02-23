# I recognize this is not really structured as a test, I just need
# to mock something up and don't want to destroy my old F5 test.
# This may be the beginning of me moving into something that is
# a little more "TDD" like. Not full unit tests, but something more
# stable and open to automation.
#%%
from pathlib import Path
import pickle

from calicam.cameras.camera_array import CameraArray
from calicam.cameras.camera_array_builder import CameraArrayBuilder
from calicam.calibration.capture_volume.point_estimates import (
    PointEstimates,
    get_point_history,
)

from calicam.cameras.synchronizer import Synchronizer
from calicam.triangulate.stereo_points_builder import StereoPointsBuilder   
from calicam.calibration.charuco import Charuco
from calicam.recording.recorded_stream import RecordedStream, RecordedStreamPool
from calicam.triangulate.array_triangulator import ArrayTriangulator

from calicam import __root__

session_directory = Path(__root__, "tests", "5_cameras")
config_path = Path(session_directory, "config.toml")

recording_directory = Path(session_directory, "recording")

points_csv_path = Path(recording_directory, "stereotriangulated_points.csv")
point_history = get_point_history(points_csv_path)

array_builder = CameraArrayBuilder(config_path)
camera_array:CameraArray = array_builder.get_camera_array()

print(f"Optimizing initial camera array configuration ")

# camera_array.optimize(point_data, output_path = points_csv_path.parent)
camera_array.bundle_adjust(point_history, output_path=points_csv_path.parent)
camera_array.update_extrinsic_params()
# %%
#### This cell was used to create the post BA stereotriangulated pairs 
#### this was more for investigation purposes and confirming that 
#### `update_extrinsic_params` was working well. I'm going to now just move
#### forward with pulling xyz data from the least_squares_result of the camera array
#### and then convert that to a csv format that the vizualizer can read from


# Build streams from pre-recorded video
# ports = [0, 1, 2, 3, 4]
# charuco = Charuco(
#     4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
# )
# recorded_stream_pool = RecordedStreamPool(
#     ports, recording_directory, charuco=charuco
# )

# synchronize videos
# syncr = Synchronizer(recorded_stream_pool.streams, fps_target=100)
# recorded_stream_pool.play_videos()

# create a commmon point finder to grab charuco corners shared between the pair of ports
# point_stream = PairedPointStream(synchronizer=syncr)

# Build triangulator
# Note that this will automatically create the summarized output of the projected points
# this is just a temporary setup while I try to figure out something more suitable long-term

# output_file = Path(recording_directory, "triangulated_points_post_BA.csv")
# array_triangulator = ArrayTriangulator(camera_array, point_stream, output_file)



# %%

array_points_error_path = Path(recording_directory, "after_bund_adj.pkl")

with open(array_points_error_path, "rb") as file:
    array_points_error = pickle.load(file)

# Save out the 3d point data to a csv file
#%%
import pandas as pd
summary_df: pd.DataFrame = array_points_error.get_summary_df("after")
# %%
summary_df.to_csv(Path(recording_directory,"array_points_error_summary.csv"))
# %%
point3d_post_bund_adj = (summary_df
                         .filter([ "sync_index","charuco_id", "obj_x", "obj_y", "obj_z"])
                         .groupby(["sync_index", "charuco_id"])
                         .agg("mean")
                         .reset_index()
                         .rename({"charuco_id":"id", 
                                  "obj_x":"x_pos",
                                  "obj_y":"y_pos",
                                  "obj_z":"z_pos"}, axis = 1)
)

point3d_post_bund_adj["pair"] = "(0,0)"

point3d_post_bund_adj.to_csv(Path(recording_directory, "bund_adj_points_for_vizualizer.csv"))
# %%
