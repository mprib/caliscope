from pathlib import Path
from queue import Queue

from pyxy3d import __root__
from pyxy3d.helper import copy_contents
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.recording.recorded_stream import RecordedStream
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)
def test_intrinsic_calibrator():
    
    # use a general video file with a charuco for convenience
    original_data_path= Path(__root__, "tests", "sessions", "4_cam_recording")
    destination_path =Path(__root__, "tests", "sessions_copy_delete", "4_cam_recording")
    copy_contents(original_data_path,destination_path)


    recording_directory = Path(
        __root__, "tests", "sessions", "post_monocal", "calibration", "extrinsic"
    )

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )

    charuco_tracker = CharucoTracker(charuco)
    
    stream  = RecordedStream(recording_directory,port=1,rotation_count=0, tracker=charuco_tracker)

    frame_q = Queue()
    stream.subscribe(frame_q)
    
    stream.play_video()
    stream.pause()
    while frame_q.qsize() > 0:
        packet = frame_q.get() # pull off frame 0 to clear queue

    stream.jump_to(3)
    packet = frame_q.get()
    logger.info(packet.frame_index)

    stream.jump_to(7)
    packet = frame_q.get()
    logger.info(packet.frame_index)
    
    stream.jump_to(17)
    packet = frame_q.get()
    logger.info(packet.frame_index)

    
if __name__ == "__main__":
    test_intrinsic_calibrator()
    
