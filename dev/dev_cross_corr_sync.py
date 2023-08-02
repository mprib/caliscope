from pyxy3d.logger import get

logger = get(__name__)

from pathlib import Path

from pyxy3d.trackers.tracker_enum import TrackerEnum
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.calibration.charuco import Charuco

# first things first, need to process the .mp4 files and create individual files with their tracked data.

recording_directory = Path(
    r"C:\Users\Mac Prible\OneDrive\pyxy3d\test_record\recording_1"
)

mp4s = recording_directory.glob("*.mp4")

charuco = Charuco(
    columns=4,
    rows=5,
    board_height=11,
    board_width=8.5,
    dictionary="DICT_4X4_50",
    units="inch",
    aruco_scale=0.75,
    square_size_overide_cm=5.4,
    inverted=True,
)

charuco_tracker = CharucoTracker(charuco)
video_streams = []

for file in mp4s:
    logger.info(f"Processing {file.name}")
    video_stream = RecordedStream()