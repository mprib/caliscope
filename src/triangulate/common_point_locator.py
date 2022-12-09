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


class PairedPointsLocator:
    def __init__(self, synchronizer, finder, pairs):

        self.bundle_in_q = Queue()
        self.synchronizer = synchronizer
        self.synchronizer.subscribe_to_bundle(self.bundle_in_q)

        # finder must have a function called "find_points" which takes a
        # frame as the single argument and returns a list of ids and a img_locations
        self.finder = finder

        self.paired_points_q = Queue()
        self.pairs = pairs

        self.thread = Thread(target=self.find_paired_points, args=[], daemon=False)
        self.thread.start()

    def find_paired_points(self):

        while self.synchronizer.continue_synchronizing:
            bundle = self.bundle_in_q.get()

            points = {}  # will be populated with dataframes of: id | x | y
            for port in bundle.keys():
                if bundle[port] is not None:
                    frame = bundle[port]["frame"]
                    ids, loc_img = self.finder.find_points(frame)
                    if ids.any():
                        points[port] = pd.DataFrame(
                            {
                                "ids": ids,
                                "loc_img_x": loc_img[:, 0],
                                "loc_img_y": loc_img[:, 1],
                            }
                        )
                        logging.debug(f"Port: {port}: \n {points[port]}")
                        # logging.debug(f"Port: {port}: ids: {ids}, loc: {loc_img}")
                        # data = np.hstack(ids.squeeze(), loc_img.squeeze())

            for pair in self.pairs:
                if pair[0] in points.keys() and pair[1] in points.keys():
                    print("Entering inner join loop")
                    common_points = points[pair[0]].merge(
                        points[pair[1]],
                        on="ids",
                        how="inner",
                        suffixes=[f"_{pair[0]}",f"_{pair[1]}"]
                    )
                    logging.debug(f"Points in common for ports {pair}: \n {common_points}")


if __name__ == "__main__":
    from src.recording.recorded_stream import RecordedStreamPool

    from src.calibration.charuco import Charuco
    import time

    repo = Path(__file__).parent.parent.parent
    print(repo)
    video_directory = Path(
        repo, "src", "triangulate", "sample_data", "stereo_track_charuco"
    )

    ports = [0, 1]
    recorded_stream_pool = RecordedStreamPool(ports, video_directory)
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
    recorded_stream_pool.play_videos()

    notification_q = Queue()  # used internally here just to display frames
    syncr.subscribe_to_notice(notification_q)

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide=0.0525, inverted=True
    )

    trackr = CornerTracker(charuco)
    pairs = [(0, 1)]
    locatr = PairedPointsLocator(
        synchronizer=syncr,
        finder=trackr,
        pairs=pairs,
    )

    while syncr.continue_synchronizing:
        frame_bundle_notice = notification_q.get()
        for port, frame_data in syncr.current_bundle.items():
            if frame_data:
                cv2.imshow(f"Port {port}", frame_data["frame"])

        key = cv2.waitKey(1)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break
