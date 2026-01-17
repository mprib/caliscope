import logging
from pathlib import Path
from queue import Empty, Queue
from time import sleep, time

import numpy as np

from caliscope import __root__
from caliscope.core.charuco import Charuco
from caliscope.core.intrinsic_calibrator import IntrinsicCalibrator
from caliscope.cameras.camera_array import CameraData
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.recording import create_streamer
from caliscope.trackers.charuco_tracker import CharucoTracker

logger = logging.getLogger(__name__)


def test_intrinsic_calibrator(tmp_path: Path):
    # use a general video file with a charuco for convenience
    original_data_path = Path(__root__, "tests", "sessions", "4_cam_recording")
    copy_contents_to_clean_dest(original_data_path, tmp_path)

    recording_directory = Path(__root__, "tests", "sessions", "post_monocal", "calibration", "extrinsic")

    charuco = Charuco(4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True)

    charuco_tracker = CharucoTracker(charuco)

    streamer = create_streamer(
        video_directory=recording_directory,
        port=1,
        tracker=charuco_tracker,
    )

    camera = CameraData(port=0, size=streamer.size)  # fresh camera for calibration

    assert camera.rotation is None
    assert camera.translation is None
    assert camera.matrix is None
    assert camera.distortions is None

    intrinsic_calibrator = IntrinsicCalibrator(camera, streamer)

    frame_q = Queue()
    streamer.subscribe(frame_q)

    streamer.start()
    streamer.pause()

    # Drain any initial frames - use timeout-based drain since qsize() is unreliable
    # in multithreaded contexts
    while True:
        try:
            frame_q.get(timeout=0.2)
        except Empty:
            break

    test_frames = [3, 5, 7, 9, 20, 25]
    for i in test_frames:
        streamer.seek_to(i)
        # Get packets until we find the one from the seek.
        # Due to race conditions, there may be stale packets from before pause took
        # full effect. The seek will produce exactly one packet with the target index.
        timeout_at = time() + 2.0
        while time() < timeout_at:
            packet = frame_q.get(timeout=0.5)
            if packet.frame_index == i:
                break
        assert i == packet.frame_index, f"Expected frame {i}, got {packet.frame_index}"
        logger.info(packet.frame_index)
        intrinsic_calibrator.add_frame_packet(packet)
        intrinsic_calibrator.add_calibration_frame_index(packet.frame_index)

    streamer.unpause()
    streamer.stop()
    intrinsic_calibrator.stop_event.set()

    logger.info(camera.get_display_data())

    intrinsic_calibrator.calibrate_camera()
    logger.info(camera)

    # basic assertions to confirm return values from opencv calibration
    assert camera.grid_count == 6
    assert isinstance(camera.matrix, np.ndarray)
    assert isinstance(camera.distortions, np.ndarray)
    assert camera.error is not None and camera.error > 0

    logger.info(camera.get_display_data())


def test_autopopulate_data(tmp_path: Path):
    # use a general video file with a charuco for convenience
    original_data_path = Path(__root__, "tests", "sessions", "4_cam_recording")
    copy_contents_to_clean_dest(original_data_path, tmp_path)

    recording_directory = Path(__root__, "tests", "sessions", "post_monocal", "calibration", "extrinsic")

    charuco = Charuco(4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True)

    charuco_tracker = CharucoTracker(charuco)

    streamer = create_streamer(
        video_directory=recording_directory,
        port=1,
        tracker=charuco_tracker,
        fps_target=100,  # Fast playback for autopopulation
    )

    camera = CameraData(port=0, size=streamer.size)  # fresh camera for calibration

    assert camera.rotation is None
    assert camera.translation is None
    assert camera.matrix is None
    assert camera.distortions is None

    intrinsic_calibrator = IntrinsicCalibrator(camera, streamer)

    # handy way to peek into what is going on
    frame_q = Queue()
    streamer.subscribe(frame_q)
    streamer.start()
    streamer.pause()

    _ = frame_q.get()  # pull off frame 0 to clear queue

    target_grid_count = 25
    wait_between = 3
    threshold_corner_count = 6
    intrinsic_calibrator.initiate_auto_pop(
        wait_between=wait_between,
        threshold_corner_count=threshold_corner_count,
        target_grid_count=target_grid_count,
    )

    streamer.seek_to(0)
    streamer.unpause()

    while intrinsic_calibrator.auto_store_data.is_set():
        actual_grid_count = len(intrinsic_calibrator.calibration_frame_indices)
        logger.info(f"waiting for data to populate...currently {actual_grid_count}")

        sleep(0.5)

    intrinsic_calibrator.backfill_calibration_frames()
    intrinsic_calibrator.calibrate_camera()
    assert camera.grid_count == target_grid_count
    logger.info(f"Calibration complete: {camera}")


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_intrinsic_calibrator(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_autopopulate_data(Path(tmp))
