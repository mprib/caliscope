from pyxy3d.logger import get
import cv2

logger = get(__name__)

from pathlib import Path
from moviepy.editor import VideoFileClip
from pyxy3d.trackers.tracker_enum import TrackerEnum
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.interface import FramePacket

# first things first, need to process the .mp4 files and create individual files with their tracked data.


def get_video_data(file_path):
    video_data = {}
    with VideoFileClip(str(file_path)) as clip:
        video_data["size"] = clip.size
        video_data["fps"] = clip.fps
        video_data["start"] = clip.start
        video_data["end"] = clip.end
    return video_data


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

camera_index = 0
rotation_count = 0  # none of the files are rotated...
# resolutions = get_resolutions_of_all_mp4s_in_folder(recording_directory)

for file_path in mp4s:
    logger.info(f"Processing {file_path.name}")

    frame_index = 0
    file_data = get_video_data(file_path)
    start_time = file_data["start"]
    fps = file_data["fps"]

    logger.info(f"For path:{file_path} the data is {file_data}")

    capture = cv2.VideoCapture(str(file_path))
    success = True
    while True:
        success, frame = capture.read()

        if not success:
            break

        point_packet = charuco_tracker.get_points(frame, camera_index, rotation_count)
        frame_packet = FramePacket(
            camera_index,
            frame_time=frame_index,
            frame=frame,
            points=point_packet,
            draw_instructions=charuco_tracker.draw_instructions,
        )
        frame_index += 1

        cv2.imshow(f"Port {camera_index}", frame_packet.frame_with_points)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    camera_index += 1
