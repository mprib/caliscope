from pathlib import Path
import sys
from threading import Thread

repo = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo))

from src.recording.recorded_stream import RecordedStreamPool
from src.cameras.synchronizer import Synchronizer
from src.calibration.charuco import Charuco
from triangulate.paired_point_stream import PairedPointStream
from src.triangulate.stereo_triangulator import StereoTriangulator
from src.triangulate.visualization.stereo_visualizer import StereoVisualizer
from src.calibration.corner_tracker import CornerTracker


# set the location for the sample data used for testing
video_directory = Path(
    repo, "src", "triangulate", "sample_data", "stereo_track_charuco"
)
# create playback streams to provide to synchronizer
ports = [0, 1]
recorded_stream_pool = RecordedStreamPool(ports, video_directory)
syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
recorded_stream_pool.play_videos()
# create a corner tracker to locate board corners
charuco = Charuco(
    4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide=0.0525, inverted=True
)
trackr = CornerTracker(charuco)
# create a commmon point finder to grab charuco corners shared between the pair of ports
pairs = [(0, 1)]
locatr = PairedPointStream(
    synchronizer=syncr,
    pairs=pairs,
    tracker=trackr,
)
sample_config_path = str(Path(video_directory.parent, "config.toml"))
triangulatr = StereoTriangulator(0, 1, sample_config_path)


# Create a seperate thread for the visualizer to run in so it doesn't block

vizr = StereoVisualizer(triangulatr)

thread = Thread(target=StereoVisualizer)


while True:
    common_points = locatr.out_q.get()
    for index, row in common_points.iterrows():
        point_A = (row["loc_img_x_0"], row["loc_img_y_0"])
        point_B = (row["loc_img_x_1"], row["loc_img_y_1"])
        print(triangulatr.triangulate(point_A, point_B))
