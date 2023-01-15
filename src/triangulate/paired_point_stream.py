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
from dataclasses import dataclass

from src.cameras.synchronizer import Synchronizer
from src.calibration.corner_tracker import CornerTracker


class PairedPointStream:
    def __init__(self, synchronizer, pairs, tracker, csv_output_path=None):

        self.synched_frames_in_q = Queue(-1)
        self.synchronizer = synchronizer
        self.synchronizer.subscribe_to_synched_frames(self.synched_frames_in_q)

        self.tracker = tracker  # this is just for charuco tracking...will need to expand on this for mediapipe later

        self.out_q = Queue(-1)  # no size limitations...should be small data
        self.pairs = pairs

        self.csv_output_path = csv_output_path
        self.tidy_output = {}  # a holding place for data to be saved to csv

        self.thread = Thread(target=self.find_paired_points, args=[], daemon=True)
        self.thread.start()

    def add_to_tidy_output(self, packet):
        """
        Convert the packet to a dictionary and add it to a running dict of lists
        Creates something that can be quickly exported to csv
        """
        if self.csv_output_path is None:
            return

        tidy_packet = packet.to_dict()
        if len(self.tidy_output) == 0:
            for key, value in tidy_packet.copy().items():
                self.tidy_output[key] = value
        else:
            for key, value in tidy_packet.copy().items():
                self.tidy_output[key].extend(value)

    def get_paired_points_packet(self, pair, points_df):
        #TODO: #54 refactor paired point finder to use numpy rather than relying on pandas        
        paired_points = points_df[pair[0]].merge(
            points_df[pair[1]],
            on="ids",
            how="inner",
            suffixes=[f"_A", f"_B"],
        )
        port_A = pair[0]
        port_B = pair[1]

        time_A = points_df[port_A]["frame_time"][0]
        time_B = points_df[port_B]["frame_time"][0]

        sync_index = points_df[port_A]["sync_index"][0]

        point_id = np.array(paired_points["ids"], dtype=np.int64)

        loc_img_x_A = np.array(paired_points["loc_img_x_A"], dtype=np.float64)
        loc_img_x_B = np.array(paired_points["loc_img_x_B"], dtype=np.float64)

        loc_img_y_A = np.array(paired_points["loc_img_y_A"], dtype=np.float64)
        loc_img_y_B = np.array(paired_points["loc_img_y_B"], dtype=np.float64)

        loc_board_x = np.array(paired_points["loc_board_x_A"], dtype=np.float64)
        loc_board_y = np.array(paired_points["loc_board_y_A"], dtype=np.float64)

        if len(point_id) == 0:
            packet = None
        else:
            packet = PairedPointsPacket(
                sync_index=sync_index,
                port_A=port_A,
                port_B=port_B,
                time_A=time_A,
                time_B=time_B,
                point_id=point_id,
                loc_board_x=loc_board_x,
                loc_board_y=loc_board_y,
                loc_img_x_A=loc_img_x_A,
                loc_img_y_A=loc_img_y_A,
                loc_img_x_B=loc_img_x_B,
                loc_img_y_B=loc_img_y_B,
            )
            logging.debug(f"Points in common for ports {pair}: \n {paired_points}")
        return packet
        
        
        
    def find_paired_points(self):
        while True:
            synched_frames = self.synched_frames_in_q.get()

            # will be populated with dataframes of:
            # id | img_x | img_y | board_x | board_y
            points = {}

            # find points in each of the frames
            for port in synched_frames.keys():
                if synched_frames[port] is not None:
                    frame = synched_frames[port]["frame"]
                    frame_time = synched_frames[port]["frame_time"]
                    sync_index = synched_frames[port]["sync_index"]

                    ids, loc_img, loc_board = self.tracker.get_corners(frame)
                    if ids.any():
                        points[port] = pd.DataFrame(
                            {
                                "frame_time": frame_time,
                                "sync_index": sync_index,
                                "ids": ids[:, 0].tolist(),
                                "loc_img_x": loc_img[:, 0][:, 0].tolist(),
                                "loc_img_y": loc_img[:, 0][:, 1].tolist(),
                                "loc_board_x": loc_board[:, 0][:, 0].tolist(),
                                "loc_board_y": loc_board[:, 0][:, 1].tolist(),
                            }
                        )
                        logging.debug(f"Port: {port}: \n {points[port]}")
                        
            # paired_points = None
            for pair in self.pairs:
                if pair[0] in points.keys() and pair[1] in points.keys():

                    packet = self.get_paired_points_packet(pair, points)
                    
                    # if no points in common, then don't do anything
                    if packet is None:
                        pass
                    else:
                        self.out_q.put(packet)
                        self.add_to_tidy_output(packet)


@dataclass
class PairedPointsPacket:
    sync_index: int

    port_A: int
    port_B: int

    time_A: float
    time_B: float

    point_id: np.ndarray

    loc_board_x: np.ndarray
    loc_board_y: np.ndarray

    loc_img_x_A: np.ndarray
    loc_img_y_A: np.ndarray

    loc_img_x_B: np.ndarray
    loc_img_y_B: np.ndarray

    @property
    def pair(self):
        return (self.port_A, self.port_B)

    def to_dict(self):
        """A method that can be used by the caller of the point stream to create a
        dictionary of lists that is well-formatted to be turned into a tidy dataframe
        for export to csv."""

        packet_dict = {}
        length = len(self.point_id)
        packet_dict["sync_index"] = [self.sync_index] * length
        packet_dict["port_A"] = [self.port_A] * length
        packet_dict["port_B"] = [self.port_B] * length
        packet_dict["time_A"] = [self.time_A] * length
        packet_dict["time_B"] = [self.time_B] * length
        packet_dict["point_id"] = self.point_id.tolist()
        packet_dict["loc_board_x"] = self.loc_board_x.tolist()
        packet_dict["loc_board_y"] = self.loc_board_y.tolist()
        packet_dict["loc_img_x_A"] = self.loc_img_x_A.tolist()
        packet_dict["loc_img_y_A"] = self.loc_img_y_A.tolist()
        packet_dict["loc_img_x_B"] = self.loc_img_x_B.tolist()
        packet_dict["loc_img_y_B"] = self.loc_img_y_B.tolist()

        return packet_dict


if __name__ == "__main__":
    from src.recording.recorded_stream import RecordedStreamPool
    from src.calibration.charuco import Charuco

    repo = Path(__file__).parent.parent.parent
    print(repo)
    session_directory = Path(repo, "sessions", "iterative_adjustment", "recording")
    csv_output = Path(session_directory, "paired_point_data.csv")

    ports = [0, 1, 2]
    recorded_stream_pool = RecordedStreamPool(ports, session_directory)
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
    recorded_stream_pool.play_videos()

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )

    trackr = CornerTracker(charuco)

    pairs = [(0, 1), (0, 2), (1, 2)]

    point_stream = PairedPointStream(
        synchronizer=syncr, pairs=pairs, tracker=trackr, csv_output_path=csv_output
    )

    while True:
        points_packet = point_stream.out_q.get()

        # print("--------------------------------------")
        # print(points_packet)
        
        if points_packet.sync_index ==300:
            save_data = pd.DataFrame(point_stream.tidy_output)
            save_data.to_csv(csv_output)
