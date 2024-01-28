from pathlib import Path
import random    
from queue import Queue
from time import sleep
import numpy as np
from caliscope import __root__
from caliscope.helper import copy_contents
from caliscope.calibration.charuco import Charuco
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.recording.recorded_stream import RecordedStream
from caliscope.cameras.camera_array import CameraData
import caliscope.logger

from caliscope.calibration.intrinsic_calibrator import IntrinsicCalibrator

logger = caliscope.logger.get(__name__)


def test_intrinsic_calibrator():
    # use a general video file with a charuco for convenience
    original_data_path = Path(__root__, "tests", "sessions", "4_cam_recording")
    destination_path = Path(
        __root__, "tests", "sessions_copy_delete", "4_cam_recording"
    )
    copy_contents(original_data_path, destination_path)

    recording_directory = Path(
        __root__, "tests", "sessions", "post_monocal", "calibration", "extrinsic"
    )

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )

    charuco_tracker = CharucoTracker(charuco)

    stream = RecordedStream(
        recording_directory, port=1, rotation_count=0, tracker=charuco_tracker
    )

    camera = CameraData(port=0, size=stream.size)

    assert camera.rotation is None
    assert camera.translation is None
    assert camera.matrix is None
    assert camera.distortions is None

    intrinsic_calibrator = IntrinsicCalibrator(camera, stream)

    frame_q = Queue()
    stream.subscribe(frame_q)

    stream.play_video()
    stream.pause()

    packet = frame_q.get()  # pull off frame 0 to clear queue

    # safety check to really clear queue
    while frame_q.qsize() > 0:
        packet = frame_q.get()

    test_frames = [3, 5, 7, 9, 20, 25]
    for i in test_frames:
        stream.jump_to(i)
        packet = frame_q.get()
        assert i == packet.frame_index
        logger.info(packet.frame_index)
        intrinsic_calibrator.add_frame_packet(packet)
        intrinsic_calibrator.add_calibration_frame_index(packet.frame_index)

    stream.stop_event.set()
    stream.unpause()
    intrinsic_calibrator.stop_event.set()

    logger.info(camera.get_display_data())

    intrinsic_calibrator.calibrate_camera()
    logger.info(camera)

    # basic assertions to confirm return values from opencv calibration
    assert camera.grid_count == 6
    assert isinstance(camera.matrix, np.ndarray)
    assert isinstance(camera.distortions, np.ndarray)
    assert camera.error > 0

    logger.info(camera.get_display_data())


def test_autopopulate_data():
    # use a general video file with a charuco for convenience
    original_data_path = Path(__root__, "tests", "sessions", "4_cam_recording")
    destination_path = Path(
        __root__, "tests", "sessions_copy_delete", "4_cam_recording"
    )
    copy_contents(original_data_path, destination_path)

    recording_directory = Path(
        __root__, "tests", "sessions", "post_monocal", "calibration", "extrinsic"
    )

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )

    charuco_tracker = CharucoTracker(charuco)

    stream = RecordedStream(
        recording_directory, port=1, rotation_count=0, tracker=charuco_tracker
    )

    camera = CameraData(port=0, size=stream.size)

    assert camera.rotation is None
    assert camera.translation is None
    assert camera.matrix is None
    assert camera.distortions is None

    intrinsic_calibrator = IntrinsicCalibrator(camera, stream)

    # handy way to peek into what is going on
    frame_q = Queue()
    stream.subscribe(frame_q)
    stream.set_fps_target(100)
    stream.play_video()
    stream.pause()

    packet = frame_q.get()  # pull off frame 0 to clear queue

    target_grid_count = 25
    wait_between = 3
    threshold_corner_count = 6
    intrinsic_calibrator.initiate_auto_pop(
        wait_between=wait_between,
        threshold_corner_count=threshold_corner_count,
        target_grid_count=target_grid_count,
    )

    stream.jump_to(0)
    stream.unpause()
    
    while intrinsic_calibrator.auto_store_data.is_set():
        actual_grid_count = len(intrinsic_calibrator.calibration_frame_indices)
        logger.info(f"waiting for data to populate...currently {actual_grid_count}")
        
        sleep(.5)

    # actual_grid_count = len(intrinsic_calibrator.calibration_frame_indices)
    # # build new frame list
    # new_potential_frames = []
    # for frame_index, ids in intrinsic_calibrator.all_ids.items():
    #     if frame_index not in intrinsic_calibrator.calibration_frame_indices:
    #         if len(ids) > 3: # just a quick check for minimal data in the frame
    #             new_potential_frames.append(frame_index)
            
    # sample_size = target_grid_count-actual_grid_count
    # sample_size = min(sample_size, len(new_potential_frames))
    # sample_size = max(sample_size,0)

    # random_frames = random.sample(new_potential_frames,sample_size)
    # for frame in random_frames:
    #     intrinsic_calibrator.add_calibration_frame_index(frame)

    intrinsic_calibrator.backfill_calibration_frames()
    intrinsic_calibrator.calibrate_camera()
    assert(camera.grid_count ==target_grid_count)
    logger.info(f"Calibration complete: {camera}")

if __name__ == "__main__":

    test_intrinsic_calibrator()
    test_autopopulate_data()
