from pyxy3d.logger import get
import cv2
from threading import Thread

logger = get(__name__)
import pandas as pd
import numpy as np
from pathlib import Path
from moviepy.editor import VideoFileClip
from pyxy3d.trackers.tracker_enum import TrackerEnum
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.interface import FramePacket, Tracker
# first things first, need to process the .mp4 files and create individual files with their tracked data.


def get_video_data(file_path):
    video_data = {}
    with VideoFileClip(str(file_path)) as clip:
        video_data["size"] = clip.size
        video_data["fps"] = clip.fps
        video_data["start"] = clip.start
        video_data["end"] = clip.end
    return video_data

def save_point_data(mp4_path:Path, tracker:Tracker, camera_index:int, rotation_count:int):

    frame_index = 0
    file_data = get_video_data(mp4_path)
    start_time = file_data["start"]
    fps = file_data["fps"]

    logger.info(f"For path:{mp4_path} the data is {file_data}")

    capture = cv2.VideoCapture(str(mp4_path))
    point_data = {"sync_index":[],
                  "port":[],
                  "frame_time":[],
                  "point_id":[],
                  "img_loc_x":[],
                  "img_loc_y":[],
                  "obj_loc_x":[],
                  "obj_loc_y":[]}  
    while True:
        success, frame = capture.read()

        if not success:
            break

        point_packet = tracker.get_points(frame, camera_index, rotation_count)
        frame_packet = FramePacket(
            camera_index,
            frame_time=frame_index,
            frame=frame,
            points=point_packet,
            draw_instructions=tracker.draw_instructions,
        )
        
        new_tidy_table = frame_packet.to_tidy_table(frame_index)
        if new_tidy_table is not None:  # i.e. it has data
            for key, value in point_data.copy().items():
                logger.debug("Extending tidy table of point history")
                point_data[key].extend(new_tidy_table[key])

        frame_index += 1

    point_data_path = Path(mp4_path.parent, f"point_data_{camera_index}.csv")
    logger.info(f"Saving out point data for video file associated with camera {camera_index}...")
    pd.DataFrame(point_data).to_csv(point_data_path)
    

 
 
def create_points_in_directory(recording_directory:Path, tracker:Tracker):

    mp4s = recording_directory.glob("*.mp4")

    video_streams = []

    camera_index = 0
    rotation_count = 0  # none of the files are rotated...
    # resolutions = get_resolutions_of_all_mp4s_in_folder(recording_directory)

    for mp4_path in mp4s:
        logger.info(f"Begin processing of {mp4_path.name}")

        thread = Thread(target=save_point_data, args = [mp4_path, tracker, camera_index, rotation_count])
        thread.start()
        camera_index += 1


def remove_random_frames(file_path, fps=30, seed=42):
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Load data
    data = pd.read_csv(file_path)
    
    # Calculate the number of frames equivalent to 2 seconds
    max_frames_to_remove = 2 * fps
    
    # Generate random number of frames to remove from the beginning and end
    frames_to_remove_start = np.random.randint(0, max_frames_to_remove)
    frames_to_remove_end = np.random.randint(0, max_frames_to_remove)
    
    min_frame = data["sync_index"].min()
    max_frame = data["sync_index"].max()
    
    
    # Remove frames
    data = data.query(f"sync_index > {min_frame + frames_to_remove_start} and sync_index < {max_frame-frames_to_remove_end}")
    
    # Reset index
    data.reset_index(drop=True, inplace=True)
    
    # Generate new file path
    file_path = Path(file_path)
    new_file_path = file_path.with_name(file_path.stem + "_alt" + file_path.suffix)
    
    # Save to new file
    data.to_csv(new_file_path, index=False)
    

if __name__ == "__main__":
    recording_directory = Path(
        r"C:\Users\Mac Prible\OneDrive\pyxy3d\test_record\recording_1"
    )

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

    tracker = CharucoTracker(charuco)

    # comment this out so you don't have to rerun it every time    
    # create_points_in_directory(recording_directory, tracker)

    for csv_path in recording_directory.glob("*_alt.csv"):
        logger.info(f"removing file contained at {csv_path}")
        csv_path.unlink()

    for csv_path in recording_directory.glob("point_data_*"):
        logger.info(f"Creating alternate data for test purposes with beginning and ending data deleted")
        remove_random_frames(csv_path)
