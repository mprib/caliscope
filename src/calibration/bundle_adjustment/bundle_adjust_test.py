#%%
from pathlib import Path
import pickle
import sys

repo = str(Path(__file__)).split("src")[0]
sys.path.insert(0, repo)
print(repo)

from src.recording.recorded_stream import RecordedStreamPool
from src.cameras.synchronizer import Synchronizer
from src.cameras.camera_array_builder import CameraArrayBuilder
from src.calibration.charuco import Charuco
from src.calibration.corner_tracker import CornerTracker
from src.calibration.bundle_adjustment.bundle_adjust_functions import *
from src.calibration.bundle_adjustment.point_data import get_point_data, PointData

from src.triangulate.paired_point_stream import PairedPointStream
from src.triangulate.array_triangulator import ArrayTriangulator


n_cam_param = 6

# RERUN_POINT_TRIANGULATION = True
RERUN_POINT_TRIANGULATION = False

REFRESH_BUNDLE_ADJUST = True
# REFRESH_BUNDLE_ADJUST = False


session_directory = Path(repo, "sessions", "iterative_adjustment")
config_path = Path(session_directory, "config.toml")
array_builder = CameraArrayBuilder(config_path)
camera_array = array_builder.get_camera_array()
points_csv_path = Path(
    session_directory, "recording", "triangulated_points.csv"
)

optimized_path = Path(session_directory, "recording", "optimized_params.pkl")

for port, cam in camera_array.cameras.items():
    print(f"Port {port} translation: {cam.translation.T}")


point_data = get_point_data(points_csv_path)

optimized = bundle_adjust(camera_array, point_data)
camera_array.update_extrinsic_params(optimized.x)

for port, cam in camera_array.cameras.items():
    print(f"Port {port} translation: {cam.translation.T}")

# get the reprojection errors for each 2d
xy_reproj_error = optimized.fun.reshape(-1, 2)
euclidean_distance_error = np.sqrt(np.sum(xy_reproj_error ** 2, axis=1))
rmse_reproj_error = np.sqrt(np.mean(euclidean_distance_error**2))
print(f"Optimization run with {optimized.fun.shape[0]/2} image points")
print(f"RMSE of reprojection is {rmse_reproj_error}")


percent_cutoff = 0.9
# error_rank = np.argsort(euclidean_distance_error)
# n_2d_points = error_rank.shape[0]
# error_percent_rank = error_rank / n_2d_points

# include = error_percent_rank < percent_cutoff

point_data.filter(optimized.fun, percent_cutoff)

optimized = bundle_adjust(camera_array, point_data)

xy_reproj_error = optimized.fun.reshape(-1, 2)
euclidean_distance_error = np.sqrt(np.sum(xy_reproj_error ** 2, axis=1))
rmse_reproj_error = np.sqrt(np.mean(euclidean_distance_error**2))
print(f"Optimization run with {optimized.fun.shape[0]/2} image points")
print(f"RMSE of reprojection is {rmse_reproj_error}")

camera_array.update_extrinsic_params(optimized.x)
for port, cam in camera_array.cameras.items():
    print(f"Port {port} translation: {cam.translation.T}")

#%%
# rerun triangulation of points
if RERUN_POINT_TRIANGULATION:
    # Build streams from pre-recorded video
    recording_directory = Path(session_directory, "recording")
    ports = [0, 1, 2]
    recorded_stream_pool = RecordedStreamPool(ports, recording_directory)

    # synchronize videos
    recorded_stream_pool.play_videos()
    syncr = Synchronizer(recorded_stream_pool.streams)

    # create a corner tracker to locate board corners
    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )
    trackr = CornerTracker(charuco)

    # create a commmon point finder to grab charuco corners shared between the pair of ports
    pairs = [(0, 1), (0, 2), (1, 2)]
    point_stream = PairedPointStream(
        synchronizer=syncr,
        pairs=pairs,
        tracker=trackr,
    )

    # Build triangulator
    # Note that this will automatically create the summarized output of the projected points
    # this is just a temporary setup while I try to figure out something more suitable long-term

    output_file = Path(recording_directory, "triangulated_points_bundle_adjusted.csv")
    array_triangulator = ArrayTriangulator(camera_array, point_stream, output_file)
