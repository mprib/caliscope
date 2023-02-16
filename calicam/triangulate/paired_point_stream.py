import logging
import calicam.logger

logger = calicam.logger.get(__name__)
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

from queue import Queue
from threading import Thread, Event
import cv2
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from itertools import combinations
from calicam.cameras.synchronizer import Synchronizer
from calicam.calibration.corner_tracker import CornerTracker
from calicam.cameras.data_packets import SyncPacket, FramePacket, PointPacket


class PairedPointStream:
    def __init__(self, synchronizer, csv_output_path=None):

        self.synchronizer = synchronizer
        self.synched_frames_in_q = Queue(
            -1
        )  # receive from synchronizer...no size limit to queue
        self.synchronizer.subscribe_to_sync_packets(self.synched_frames_in_q)

        self.out_q = Queue(-1)  # no size limitations...should be small data
        self.ports = synchronizer.ports
        self.pairs = [(i, j) for i, j in combinations(self.ports, 2) if i < j]

        self.csv_output_path = csv_output_path
        self.tidy_output = {}  # a holding place for data to be saved to csv

        self.stop_event = Event()
        self.frames_complete = False
        self.thread = Thread(target=self.create_paired_points, args=[], daemon=True)
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

    def get_paired_points_packet(
        self,
        sync_index,
        port_A,
        points_A,
        port_B,
        points_B
    ):

        # get ids in common
        if len(points_A.point_id)>0 and len(points_B.point_id)>0:
            common_ids = np.intersect1d(points_A.point_id, points_B.point_id)
        else:
            common_ids = np.array([])
        
        if len(common_ids) == 0:
            packet = None
        else:
            # common_ids = common_ids[:,0]
            # for both ports, get the indices of the common ids
            sorter_A = np.argsort(points_A.point_id)
            shared_indices_A = sorter_A[
                np.searchsorted(points_A.point_id, common_ids, sorter=sorter_A)
            ]
            shared_indices_A

            sorter_B = np.argsort(points_B.point_id)
            shared_indices_B = sorter_B[
                np.searchsorted(points_B.point_id, common_ids, sorter=sorter_B)
            ]
            shared_indices_B

            packet = PairedPointsPacket(
                sync_index=sync_index,
                port_A=port_A,
                port_B=port_B,
                common_ids=common_ids,
                img_loc_A = points_A.img_loc[shared_indices_A],
                img_loc_B = points_B.img_loc[shared_indices_B]
            )

            logger.debug(f"Points in common for ports ({port_A}, {port_B}): {common_ids}")

        return packet

    def create_paired_points(self):

        while not self.frames_complete:
            synched_frames: SyncPacket = self.synched_frames_in_q.get()

            if synched_frames is None:
                logging.info(
                    "End of frames signaled...paired point stream shutting down"
                )
                self.frames_complete = True
                self.out_q.put(None)
                break

            # will be populated with dataframes of:
            # id | img_x | img_y | board_x | board_y
            sync_index = synched_frames.sync_index

            # paired_points = None
            for pair in self.pairs:
                port_A = pair[0]
                port_B = pair[1]


                if (
                    synched_frames.frame_packets[port_A] is not None
                    and synched_frames.frame_packets[port_B] is not None
                ):

                    points_A = synched_frames.frame_packets[port_A].points
                    points_B = synched_frames.frame_packets[port_B].points

                    paired_points: PairedPointsPacket = self.get_paired_points_packet(
                        sync_index, port_A, points_A, port_B, points_B
                    )

                    # if no points in common, then don't do anything
                    if paired_points is None:
                        pass
                    else:
                        self.out_q.put(paired_points)
                        logger.info(
                            f"Placing packet for sync index {paired_points.sync_index} and pair {pair}"
                        )
                        # self.add_to_tidy_output(paired_points)


@dataclass
class PairedPointsPacket:
    """The points shared by two FramePointsPackets"""
    sync_index: int

    port_A: int
    port_B: int

    common_ids: np.ndarray
    img_loc_A: np.ndarray
    img_loc_B: np.ndarray

    @property
    def pair(self):
        return (self.port_A, self.port_B)


if __name__ == "__main__":
    from calicam.recording.recorded_stream import RecordedStreamPool
    from calicam.calibration.charuco import Charuco

    logger.setLevel(logging.DEBUG)
    from calicam import __root__

    session_directory = Path(__root__, "tests", "5_cameras", "recording")
    csv_output = Path(session_directory, "paired_point_data.csv")

    ports = [0, 1, 2, 3, 4]

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )

    recorded_stream_pool = RecordedStreamPool(ports, session_directory, charuco=charuco)
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=200)
    recorded_stream_pool.play_videos()

    point_stream = PairedPointStream(synchronizer=syncr, csv_output_path=csv_output)

    # I think that EOF needs to propogate up
    while not point_stream.frames_complete:
        points_packet = point_stream.out_q.get()

        # print("--------------------------------------")
        # print(points_packet)

    print("Saving data....")
    save_data = pd.DataFrame(point_stream.tidy_output)
    save_data.to_csv(csv_output)
