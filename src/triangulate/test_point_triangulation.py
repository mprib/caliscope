from pathlib import Path
import sys
from threading import Thread
import numpy as np

repo = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo))

from src.recording.recorded_stream import RecordedStreamPool
from src.cameras.synchronizer import Synchronizer
from src.calibration.charuco import Charuco
from src.triangulate.paired_point_stream import PairedPointStream
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
point_stream = PairedPointStream(
    synchronizer=syncr,
    pairs=pairs,
    tracker=trackr,
)
sample_config_path = str(Path(video_directory.parent, "config.toml"))
triangulatr = StereoTriangulator(point_stream, sample_config_path)

vizr = StereoVisualizer(triangulatr)

# while True:
all_point_3D = triangulatr.out_q.get()
vizr.add_scatter(all_point_3D)
print(all_point_3D)

vizr.start()

test_board_corners = np.array(
    [
        [-8.97169043, 8.58954463, 56.44145603],
        [-9.28875076, 4.38558929, 54.45100235],
        [-9.65045165, 0.20856289, 52.33586731],
        [-13.50551906, 9.2943572, 55.70972394],
        [-13.80153222, 5.0210905, 53.37978951],
        [-14.07910444, 0.84995609, 51.34233615],
        [-17.93996941, 9.86696736, 54.15030867],
        [-18.18285319, 5.69386964, 52.21569521],
        [-18.5111405, 1.50492472, 49.85212552],
        [-22.37649957, 10.55513425, 53.03158067],
        [-22.6366339, 6.28469088, 50.85093531],
        [-22.8193701, 2.1501825, 48.45748189],
    ]
)
vizr = StereoVisualizer(triangulatr)
vizr.add_scatter(test_board_corners)
vizr.start()

    # print(triangulatr.out_q.qsize())orts = [0, 1]