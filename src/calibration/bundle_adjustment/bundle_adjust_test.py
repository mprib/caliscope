#%%
from pathlib import Path
import pickle
import sys

repo = str(Path(__file__)).split("src")[0]
sys.path.insert(0, repo)
print(repo)

from src.recording.recorded_stream import RecordedStreamPool
from src.cameras.synchronizer import Synchronizer
from src.calibration.charuco import Charuco
from src.calibration.corner_tracker import CornerTracker
from src.calibration.bundle_adjustment.bundle_adjust_functions import *
from src.triangulate.paired_point_stream import PairedPointStream
from src.triangulate.array_triangulator import ArrayTriangulator

CAMERA_PARAM_COUNT = 6

# RERUN_POINT_TRIANGULATION = True
RERUN_POINT_TRIANGULATION = False

REFRESH_BUNDLE_ADJUST = True
# REFRESH_BUNDLE_ADJUST = False


session_directory = Path(repo, "sessions", "iterative_adjustment")
config_path = Path(session_directory, "config.toml")
array_builder = CameraArrayBuilder(config_path)
camera_array = array_builder.get_camera_array()
points_csv_path = Path(
    session_directory, "recording", "triangulated_points_daisy_chain.csv"
)

optimized_path = Path(session_directory, "recording", "optimized_params.pkl")

if REFRESH_BUNDLE_ADJUST:
    optimized = bundle_adjust(camera_array, points_csv_path)
    with open(optimized_path, "wb") as file:
        pickle.dump(optimized, file)

    print(f"RMSE: {np.sqrt(np.mean(optimized.fun**2))}")
else:
    with open(optimized_path, "rb") as file:
        optimized = pickle.load(file)


n_cameras = len(camera_array.cameras)
flat_camera_params = optimized.x[0 : n_cameras * CAMERA_PARAM_COUNT]
new_camera_params = flat_camera_params.reshape(n_cameras, CAMERA_PARAM_COUNT)
# print(new_camera_params)
xyz_points = optimized.x[n_cameras * CAMERA_PARAM_COUNT :]
point_count = int(xyz_points.shape[0] / 3)
xyz_points = xyz_points.reshape(point_count, 3)


# update camera array with new positional data
for index in range(len(new_camera_params)):
    print(index)
    port = index  # just to be explicit
    cam_vec = new_camera_params[index, :]
    camera_array.cameras[port].from_vector(cam_vec)

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
