import calicam.logger
logger = calicam.logger.get(__name__)

import time
from itertools import combinations
from pathlib import Path
from queue import Queue
from threading import Thread, Event

import cv2
import numpy as np

from calicam.cameras.synchronizer import Synchronizer


class StereoTracker:
    logger.info("Building Stereocalibrator...")

    def __init__(self, synchronizer):

        self.synchronizer = synchronizer

        self.corner_threshold = 4  # board corners in common for capture
        self.wait_time = 0.5  # seconds between snapshots
        self.grid_count_trigger = 5  #  move on to calibration

        self.synched_frames_available_q = Queue()
        self.synchronizer.subscribe_to_notice(self.synched_frames_available_q)
        self.cal_frames_ready_q = Queue()
        self.stop_event = Event()

        # build port list
        self.ports = []
        for port, stream in self.synchronizer.streams.items():
            logger.debug(f"Appending port {port}...")
            self.ports.append(port)

        # build list of pairs, but with pairs ordered (smaller, larger)
        unordered_pairs = [(i, j) for i, j in combinations(self.ports, 2)]
        self.pairs = []
        for pair in unordered_pairs:
            i, j = pair[0], pair[1]

            if i > j:
                i, j = j, i
            self.pairs.append((i, j))
            logger.debug(f"Camera pairs for calibration are: {self.pairs}")

        # Build Stereo Inputs: dictionary to hold growing lists of input parameters .
        self.stereo_inputs = {
            pair: {"common_board_loc": [], "img_loc_A": [], "img_loc_B": []}
            for pair in self.pairs
        }

        # needed to determine if enough time has passed since last capture
        self.last_corner_save_time = {pair: time.perf_counter() for pair in self.pairs}

        logger.info(f"Initiating data collection of uncalibrated pairs: {self.pairs}")
        self.keep_going = True
        self.thread = Thread(target=self.harvest_synched_frames, args=(), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.synched_frames_available_q.put("Terminate")
        logger.info("Stop signal sent in stereocalibrator")

    def harvest_synched_frames(self):
        """Monitors the synched_frames_available_q to grab a new frames and inititiate
        processing of it."""
        logger.info(
            f"Beginning to harvest corners on synched frames from port pairs: {self.pairs}"
        )

        while not self.stop_event.is_set():
            self.synched_frames_available_q.get()
            self.current_sync_packet = self.synchronizer.current_sync_packet

            if self.current_sync_packet is None:
                logger.info("Triggering stop event for stereotracker")
                self.stop_event.set()
                break
        
            logger.debug(
                "Begin determination of shared corners within current frame pairs"
            )
            for pair in self.pairs:
                self.store_stereo_data(pair)
            print("Something happening")
            self.cal_frames_ready_q.put("frames ready")

        logger.info(
            "Stereotracker synched frames harvester successfully shut-down..."
        )


    def store_stereo_data(self, pair):

        # for pair in self.uncalibrated_pairs:
        portA = pair[0]
        portB = pair[1]

        common_ids = self.get_common_ids(portA, portB)

        enough_corners = len(common_ids) > self.corner_threshold
        enough_time = (
            time.perf_counter() - self.last_corner_save_time[pair] > self.wait_time
        )

        if enough_corners and enough_time and pair in self.pairs:
            # add corner data to stereo_inputs
            obj, img_loc_A = self.get_common_locs(portA, common_ids)
            _, img_loc_B = self.get_common_locs(portB, common_ids)

            self.stereo_inputs[pair]["common_board_loc"].append(obj)
            self.stereo_inputs[pair]["img_loc_A"].append(img_loc_A)
            self.stereo_inputs[pair]["img_loc_B"].append(img_loc_B)
            self.last_corner_save_time[pair] = time.perf_counter()

    def get_common_ids(self, portA, portB):
        """Intersection of grid corners observed in the active grid pair"""
        frame_packets = self.current_sync_packet.frame_packets
        if frame_packets[portA] and frame_packets[portB]:
            ids_A = frame_packets[portA].points.point_id
            ids_B = frame_packets[portB].points.point_id
            common_ids = np.intersect1d(ids_A, ids_B)
            common_ids = common_ids.tolist()

        else:
            common_ids = []

        return common_ids

    def get_common_locs(self, port, common_ids):
        """Pull out objective location and image location of board corners for
        a port that are on the list of common ids"""

        # ids = self.current_sync_packet[port]["ids"]
        ids = self.current_sync_packet.frame_packets[port].points.point_id.tolist()
        img_loc = self.current_sync_packet.frame_packets[port].points.img_loc.tolist()
        board_loc = self.current_sync_packet.frame_packets[port].points.board_loc.tolist()

        common_img_loc = []
        common_board_loc = []

        for crnr_id, img, obj in zip(ids, img_loc, board_loc):
            if crnr_id in common_ids:
                common_board_loc.append(img)
                common_img_loc.append(obj)

        return common_img_loc, common_board_loc


if __name__ == "__main__":
    import pprint

    from calicam.calibration.corner_tracker import CornerTracker
    from calicam.session import Session
    from calicam.recording.recorded_stream import RecordedStream, RecordedStreamPool
    from calicam.calibration.charuco import Charuco
    
    logger.debug("Test live stereocalibration processing")

    repo = Path(str(Path(__file__)).split("calicam")[0], "calicam")
    recording_directory = Path(repo, "sessions", "5_cameras", "recording")

    # ports = [0, 1, 2, 3, 4]
    ports = [0,1,2]
    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )
    trackr = CornerTracker(charuco)
    recorded_stream_pool = RecordedStreamPool(ports, recording_directory, tracker=trackr)
    logger.info("Creating Synchronizer")
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
    recorded_stream_pool.play_videos()

    logger.info("Creating Stereocalibrator")
    stereo_tracker = StereoTracker(syncr)

    # while len(stereo_cal.uncalibrated_pairs) == 0:
    # time.sleep(.1)
    logger.info("Showing Stacked Frames")
    while not stereo_tracker.stop_event.is_set():

        frame_ready = stereo_tracker.cal_frames_ready_q.get()
        synched_frames = stereo_tracker.current_sync_packet.frame_packets

        for port in synched_frames.keys():
            if synched_frames[port] is not None:
                cv2.imshow(str(port), synched_frames[port].frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
    print(stereo_tracker.stereo_inputs)
