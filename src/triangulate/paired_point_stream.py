import logging

LOG_FILE = "log\common_point_locator.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)


from queue import Queue
from threading import Thread
import cv2
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.synchronizer import Synchronizer
from src.calibration.corner_tracker import CornerTracker


class PairedPointStream:
    def __init__(self, synchronizer, pairs, tracker):

        self.bundle_in_q = Queue()
        self.synchronizer = synchronizer
        self.synchronizer.subscribe_to_bundle(self.bundle_in_q)

        self.tracker = tracker  # this is just for charuco tracking...will need to expand on this for mediapipe later

        self.out_q = Queue()
        self.pairs = pairs

        self.thread = Thread(target=self.find_paired_points, args=[], daemon=True)
        self.thread.start()

    def find_paired_points(self):

        while True:
            bundle = self.bundle_in_q.get()

            points = (
                {}
            )  # will be populated with dataframes of: id | img_x | img_y | board_x | board_y

            # find points in each of the frames
            for port in bundle.keys():
                if bundle[port] is not None:
                    frame = bundle[port]["frame"]
                    frame_time = bundle[port]["frame_time"]
                    ids, loc_img, loc_board = self.tracker.get_corners(frame)
                    if ids.any():
                        points[port] = pd.DataFrame(
                            {
                                "frame_time": frame_time,
                                "ids": ids,
                                "loc_img_x": loc_img[:, 0],
                                "loc_img_y": loc_img[:, 1],
                                "loc_board_x": loc_board[:, 0],
                                "loc_board_y": loc_board[:, 1],
                            }
                        )
                        logging.debug(f"Port: {port}: \n {points[port]}")

            # create a dataframe of the shared points for each pair of frames
            for pair in self.pairs:
                if pair[0] in points.keys() and pair[1] in points.keys():
                    # print("Entering inner join loop")
                    common_points = points[pair[0]].merge(
                        points[pair[1]],
                        on="ids",
                        how="inner",
                        suffixes=[f"_{pair[0]}", f"_{pair[1]}"],
                    )
                    logging.debug(
                        f"Points in common for ports {pair}: \n {common_points}"
                    )
                    self.out_q.put(common_points)


if __name__ == "__main__":
    from src.recording.recorded_stream import RecordedStreamPool

    from src.calibration.charuco import Charuco

    repo = Path(__file__).parent.parent.parent
    print(repo)
    session_directory = Path(repo, "examples", "high_res_session")

    ports = [0, 1]
    recorded_stream_pool = RecordedStreamPool(ports, session_directory)
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
    recorded_stream_pool.play_videos()

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide=0.0525, inverted=True
    )

    trackr = CornerTracker(charuco)
    pairs = [(0, 1)]
    locatr = PairedPointStream(
        synchronizer=syncr,
        pairs=pairs,
        tracker=trackr,
    )

    while True:
        common_points = locatr.out_q.get()
        print(common_points)
