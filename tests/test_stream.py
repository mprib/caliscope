import logging
from pathlib import Path
from queue import Queue
from time import sleep

from caliscope import __root__
from caliscope.core.charuco import Charuco
from caliscope.recording import create_streamer
from caliscope.trackers.charuco_tracker import CharucoTracker

logger = logging.getLogger(__name__)


def test_streamer():
    recording_directory = Path(__root__, "tests", "sessions", "post_monocal", "calibration", "extrinsic")

    charuco = Charuco(4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True)

    charuco_tracker = CharucoTracker(charuco)

    streamer = create_streamer(
        video_directory=recording_directory,
        port=1,
        tracker=charuco_tracker,
        fps_target=6,
    )
    frame_q = Queue()
    streamer.subscribe(frame_q)
    streamer.start()
    streamer.pause()

    sleep(1)
    logger.info(f"Publisher frame index is {streamer.frame_index}")
    sleep(1)
    logger.info(f"Publisher frame index is {streamer.frame_index}")

    streamer.unpause()

    while True:
        frame_packet = frame_q.get()

        if frame_packet.frame is None:
            break

        if streamer.frame_index == 10:
            logger.info("Testing pause/unpause functionality")
            streamer.pause()
            sleep(0.5)
            test_index = streamer.frame_index
            sleep(0.5)
            # make sure that streamer doesn't advance with pause
            assert test_index == streamer.frame_index
            streamer.unpause()

        if streamer.frame_index == 15:
            logger.info("Testing ability to jump forward")
            target_frame = 20
            streamer.pause()
            sleep(1)  # need to make sure fps_target wait plays out
            streamer.seek_to(target_frame)
            sleep(1)  # need to make sure fps_target wait plays out
            # frame_index should match the jump target (the frame we just displayed)
            assert streamer.frame_index == 20

            logger.info(f"After attempting to jump to target frame {target_frame}")
            streamer.unpause()


if __name__ == "__main__":
    test_streamer()
