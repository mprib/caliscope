import logging

FILE_NAME = "log\stereocalibration.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(
    filename=FILE_NAME, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL
)

import sys
import time
from itertools import combinations
from pathlib import Path
from queue import Queue
from threading import Thread, Event

import cv2
import numpy as np

from calicam.cameras.synchronizer import Synchronizer


class StereoCalibrator:
    logging.info("Building Stereocalibrator...")

    def __init__(self, synchronizer, corner_tracker):

        self.corner_tracker = corner_tracker
        self.synchronizer = synchronizer

        self.corner_threshold = 7  # board corners in common for capture
        self.wait_time = 0.5  # seconds between snapshots
        self.grid_count_trigger = 5  #  move on to calibration


        # self.stacked_frames = Queue()  # ultimately will be removing this
        self.synched_frames_available_q = Queue()
        self.synchronizer.subscribe_to_notice(self.synched_frames_available_q)
        self.cal_frames_ready_q = Queue()
        self.stop_event = Event()

        self.build_port_list()
        self.build_uncalibrated_pairs()
        self.build_stereo_inputs()
        self.build_stereo_outputs()

        # needed to determine if enough time has passed since last capture
        self.last_corner_save_time = {
            pair: time.perf_counter() for pair in self.uncalibrated_pairs
        }

        logging.info(
            f"Initiating data collection of uncalibrated pairs: {self.uncalibrated_pairs}"
        )
        self.keep_going = True
        self.thread = Thread(target=self.harvest_synched_frames, args=(), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.synched_frames_available_q.put("Terminate")
        logging.info("Stop signal sent in stereocalibrator")
        # self.thread.join()
                
    def build_port_list(self):
        """Construct list of ports associated with incoming frames"""
        logging.debug("Creating port list...")
        self.ports = []
        for port, stream in self.synchronizer.streams.items():
            logging.debug(f"Appending port {port}...")
            self.ports.append(port)

    def build_uncalibrated_pairs(self):
        """construct a list of uncalibrated pairs with smaller number port first"""
        unordered_pairs = [(i, j) for i, j in combinations(self.ports, 2)]
        self.uncalibrated_pairs = []
        for pair in unordered_pairs:
            i, j = pair[0], pair[1]

            if i > j:
                i, j = j, i
            self.uncalibrated_pairs.append((i, j))
            logging.debug(f"Uncalibrated pairs are: {self.uncalibrated_pairs}")

        self.pairs = (
            self.uncalibrated_pairs.copy()
        )  # save original list for later reference

    def build_stereo_inputs(self):
        """Constructs dictionary to hold growing lists of input parameters .
        When a list grows to the lengths of the grid_count_trigger, it will
        commence calibration"""
        self.stereo_inputs = {
            pair: {"common_board_loc": [], "img_loc_A": [], "img_loc_B": []}
            for pair in self.pairs
        }

    def build_stereo_outputs(self):
        """Constructs dictionary to hold growing lists of input parameters .
        When a list grows to the lengths of the grid_count_trigger, it will
        commence calibration"""

        self.stereo_outputs = {
            pair: {
                "grid_count": None,
                "rotation": None,
                "translation": None,
                "RMSE": None,
            }
            for pair in self.pairs
        }

    def harvest_synched_frames(self):
        """Monitors the synched_frames_available_q to grab a new frames and inititiate
        processing of it."""
        logging.debug(f"Currently {len(self.uncalibrated_pairs)} uncalibrated pairs ")

        while not self.stop_event.set():
            self.synched_frames_available_q.get()
            
            # may get hung up on get, so additional item put on queue
            if self.stop_event.set():
                break
            
            self.current_synched_frames = self.synchronizer.current_synched_frames
            logging.debug("Synched frames harvested by stereocalibrator")

            self.add_corner_data()
            for pair in self.uncalibrated_pairs:
                self.store_stereo_data(pair)

                grid_count = len(self.stereo_inputs[pair]["common_board_loc"])
                self.stereo_outputs[pair]["grid_count"] = grid_count

                if grid_count > self.grid_count_trigger:
                    self.calibrate_thread = Thread(
                        target=self.stereo_calibrate, args=[pair], daemon=True
                    )
                    self.calibrate_thread.start()
            # self.calibrate_full_pairs()

            self.cal_frames_ready_q.put("frames ready")

            # if len(self.uncalibrated_pairs) == 0:
            #     self.stereo_calibrate()
        logging.info("Stereocalibration synched frames harvester successfully shut-down...")

    def add_corner_data(self):
        """Assign corner data for each frame"""
        for port in self.current_synched_frames.keys():
            if self.current_synched_frames[port] is not None:
                ids, img_loc, board_loc = self.corner_tracker.get_corners(
                    self.current_synched_frames[port]["frame"]
                )

                self.current_synched_frames[port]["ids"] = ids
                self.current_synched_frames[port]["img_loc"] = img_loc
                self.current_synched_frames[port]["board_loc"] = board_loc

                logging.debug(f"Port {port}: {ids}")

    def store_stereo_data(self, pair):
        logging.debug("About to process current synched frames")

        # for pair in self.uncalibrated_pairs:
        portA = pair[0]
        portB = pair[1]

        common_ids = self.get_common_ids(portA, portB)

        enough_corners = len(common_ids) > self.corner_threshold
        enough_time = (
            time.perf_counter() - self.last_corner_save_time[pair] > self.wait_time
        )

        if enough_corners and enough_time and pair in self.uncalibrated_pairs:
            # add corner data to stereo_inputs
            obj, img_loc_A = self.get_common_locs(portA, common_ids)
            _, img_loc_B = self.get_common_locs(portB, common_ids)

            self.stereo_inputs[pair]["common_board_loc"].append(obj)
            self.stereo_inputs[pair]["img_loc_A"].append(img_loc_A)
            self.stereo_inputs[pair]["img_loc_B"].append(img_loc_B)
            self.last_corner_save_time[pair] = time.perf_counter()

    def get_common_ids(self, portA, portB):
        """Intersection of grid corners observed in the active grid pair"""
        if self.current_synched_frames[portA] and self.current_synched_frames[portB]:
            ids_A = self.current_synched_frames[portA]["ids"]
            ids_B = self.current_synched_frames[portB]["ids"]
            common_ids = np.intersect1d(ids_A, ids_B)
            common_ids = common_ids.tolist()

        else:
            common_ids = []

        return common_ids

    def stereo_calibrate(self, pair):
        """Iterates across all camera pairs. Intrinsic parameters are pulled
        from camera and combined with obj and img points for each pair.
        """
        logging.info(f"About to stereocalibrate pair {pair}")

        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)

        camA = self.synchronizer.streams[pair[0]].camera
        camB = self.synchronizer.streams[pair[1]].camera

        # pull calibration parameters for pair
        obj = self.stereo_inputs[pair]["common_board_loc"]
        img_A = self.stereo_inputs[pair]["img_loc_A"]
        img_B = self.stereo_inputs[pair]["img_loc_B"]

        # convert to list of vectors for OpenCV function
        obj = [np.array(x, dtype=np.float32) for x in obj]
        img_A = [np.array(x, dtype=np.float32) for x in img_A]
        img_B = [np.array(x, dtype=np.float32) for x in img_B]

        (
            ret,
            camera_matrix_1,
            distortion_1,
            camera_matrix_2,
            distortion_2,
            rotation,
            translation,
            essential,
            fundamental,
        ) = cv2.stereoCalibrate(
            obj,
            img_A,
            img_B,
            camA.camera_matrix,
            camA.distortion,
            camB.camera_matrix,
            camB.distortion,
            imageSize=None,  # this does not matter. from OpenCV: "Size of the image used only to initialize the camera intrinsic matrices."
            criteria=criteria,
            flags=stereocalibration_flags,
        )

        self.stereo_outputs[pair] = {
            "grid_count": len(obj),
            "rotation": rotation,
            "translation": translation,
            "RMSE": ret,
        }

        if pair in self.uncalibrated_pairs:
            logging.info(f"Removing pair {pair}")
            self.uncalibrated_pairs.remove(pair)
        else:
            logging.warning(f"Attempted to remove pair {pair} but it was not present")
            
        logging.info(
            f"For camera pair {pair}, rotation is \n{rotation}\n and translation is \n{translation}"
        )
        logging.info(f"RMSE of reprojection is {ret}")


    def get_common_locs(self, port, common_ids):
        """Pull out objective location and image location of board corners for
        a port that are on the list of common ids"""

        ids = self.current_synched_frames[port]["ids"]
        img_loc = self.current_synched_frames[port]["img_loc"].tolist()
        board_loc = self.current_synched_frames[port]["board_loc"].tolist()

        common_img_loc = []
        common_board_loc = []

        for crnr_id, img, obj in zip(ids, img_loc, board_loc):
            if crnr_id in common_ids:
                common_board_loc.append(img)
                common_img_loc.append(obj)

        return common_img_loc, common_board_loc

    def reset_pair(self, pair):
        """Delete the stereo_inputs for a pair of cameras and add them back
        to the list of uncalibrated pairs"""

        self.stereo_inputs[pair]["common_board_loc"] = []
        self.stereo_inputs[pair]["img_loc_A"] = []
        self.stereo_inputs[pair]["img_loc_B"] = []

        self.stereo_outputs[pair] = {
            "grid_count": 0,
            "rotation": None,
            "translation": None,
            "RMSE": None,
        }

        if pair not in self.uncalibrated_pairs:
            self.uncalibrated_pairs.append(pair)


if __name__ == "__main__":
    import pprint

    from calicam.calibration.corner_tracker import CornerTracker
    from calicam.session import Session

    logging.debug("Test live stereocalibration processing")

    repo = Path(__file__).parent.parent.parent
    session_path = Path(repo, "sessions", "high_res_session")
    session = Session(session_path)

    session.load_cameras()
    session.load_streams()
    session.adjust_resolutions()
    # time.sleep(3)

    trackr = CornerTracker(session.charuco)

    logging.info("Creating Synchronizer")
    syncr = Synchronizer(session.streams, fps_target=6)
    logging.info("Creating Stereocalibrator")
    stereo_cal = StereoCalibrator(syncr, trackr)

    # while len(stereo_cal.uncalibrated_pairs) == 0:
    # time.sleep(.1)
    logging.info("Showing Stacked Frames")
    while len(stereo_cal.uncalibrated_pairs) > 0:

        frame_ready = stereo_cal.cal_frames_ready_q.get()
        synched_frames = stereo_cal.current_synched_frames

        for port in synched_frames.keys():
            if synched_frames[port] is not None:
                cv2.imshow(str(port), synched_frames[port]["frame"])

        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
    logging.debug(pprint.pformat(stereo_cal.stereo_inputs))
