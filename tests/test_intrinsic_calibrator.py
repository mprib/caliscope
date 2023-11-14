from pathlib import Path
from queue import Queue
import numpy as np
from pyxy3d import __root__
from pyxy3d.helper import copy_contents
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.cameras.camera_array import CameraData
import pyxy3d.logger

from pyxy3d.calibration.intrinsic_calibrator import IntrinsicCalibrator

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

    camera = CameraData(port=0,size=stream.size)

    assert(camera.rotation is None)
    assert(camera.translation is None)
    assert(camera.matrix is None)
    assert(camera.distortions is None)
    
    intrinsic_calibrator = IntrinsicCalibrator(camera, stream)

    frame_q = Queue()
    stream.subscribe(frame_q)
    
    stream.play_video()
    stream.pause()

    packet = frame_q.get() # pull off frame 0 to clear queue

    # safety check to really clear queue
    while frame_q.qsize() > 0:
        packet = frame_q.get() 

    test_frames = [3,5,7,9,20,25]
    for i in test_frames:
        stream.jump_to(i)
        packet = frame_q.get()
        assert(i == packet.frame_index)
        logger.info(packet.frame_index)
        intrinsic_calibrator.add_frame_packet(packet)
        intrinsic_calibrator.add_calibration_frame_indices(packet.frame_index)

    stream.stop_event.set()
    intrinsic_calibrator.stop_event.set()

    intrinsic_calibrator.calibrate_camera()
    logger.info(camera)

    # basic assertions to confirm return values from opencv calibration
    assert(camera.grid_count==6)
    assert(isinstance(camera.matrix, np.ndarray))
    assert(isinstance(camera.distortions, np.ndarray))
    assert(camera.error > 0)

if __name__ == "__main__":
    test_intrinsic_calibrator()
    
