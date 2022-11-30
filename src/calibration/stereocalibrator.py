import logging

FILE_NAME = "stereocalibration.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(
    filename=FILE_NAME, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL
)

import pprint
import sys
import time
from itertools import combinations
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import imutils
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.synchronizer import Synchronizer


class StereoCalibrator:
    logging.info("Building Stereocalibrator...")

    def __init__(self, synchronizer, corner_tracker):

        self.corner_tracker = corner_tracker
        self.synchronizer = synchronizer

        self.corner_threshold = 7  # board corners in common for capture
        self.wait_time = 0.5  # seconds between snapshots
        self.grid_count_trigger = 15  #  move on to calibration

        # self.stacked_frames = Queue()  # ultimately will be removing this
        self.bundle_available_q = Queue()
        self.synchronizer.subscribe(self.bundle_available_q)
        self.cal_frames_ready_q = Queue()

        self.build_port_list()
        self.build_uncalibrated_pairs()
        self.build_stereocal_inputs()

        # needed to determine if enough time has passed since last capture
        self.last_corner_save_time = {
            pair: time.perf_counter() for pair in self.uncalibrated_pairs
        }

        logging.info(
            f"Processing pairs of uncalibrated pairs: {self.uncalibrated_pairs}"
        )

        self.thread = Thread(target=self.harvest_frame_bundles, args=(), daemon=True)
        self.thread.start()

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

    def build_stereocal_inputs(self):
        """Constructs dictionary to hold growing lists of input parameters .
        When a list grows to the lengths of the grid_count_trigger, it will
        commence calibration"""
        self.stereo_inputs = {
            pair: {"common_board_loc": [], "img_loc_A": [], "img_loc_B": []}
            for pair in self.uncalibrated_pairs
        }

    def harvest_frame_bundles(self):
        """Monitors the bundle_available_q to grab a new frame bundle and inititiate
        processing of it."""
        logging.debug(f"Currently {len(self.uncalibrated_pairs)} uncalibrated pairs ")

        while len(self.uncalibrated_pairs) > 0:
            self.bundle_available_q.get()
            self.current_bundle = self.synchronizer.current_bundle

            self.add_corner_data()
            for pair in self.uncalibrated_pairs:
                self.store_stereo_data()

            self.cal_frames_ready_q.put("frames ready")

            if len(self.uncalibrated_pairs) == 0:
                self.stereo_calibrate()

    def add_corner_data(self):
        """Assign corner data for each frame"""
        for port in self.current_bundle.keys():
            if self.current_bundle[port] is not None:
                ids, img_loc, board_loc = self.corner_tracker.get_corners(
                    self.current_bundle[port]["frame"]
                )

                self.current_bundle[port]["ids"] = ids
                self.current_bundle[port]["img_loc"] = img_loc
                self.current_bundle[port]["board_loc"] = board_loc

                print(ids)
                logging.debug(f"Port {port}: {ids}")

    def store_stereo_data(self):
        logging.debug("About to process current frame bundle")

        for pair in self.uncalibrated_pairs:
            portA = pair[0]
            portB = pair[1]

            common_ids = self.get_common_ids(portA, portB)

            enough_corners = len(common_ids) > self.corner_threshold
            enough_time = (
                time.perf_counter() - self.last_corner_save_time[pair] > self.wait_time
            )

            if enough_corners and enough_time:
                # add corner data to stereo_inputs
                obj, img_loc_A = self.get_common_locs(portA, common_ids)
                _, img_loc_B = self.get_common_locs(portB, common_ids)

                self.stereo_inputs[pair]["common_board_loc"].append(obj)
                self.stereo_inputs[pair]["img_loc_A"].append(img_loc_A)
                self.stereo_inputs[pair]["img_loc_B"].append(img_loc_B)
                self.last_corner_save_time[pair] = time.perf_counter()

    def get_common_ids(self, portA, portB):
        """Intersection of grid corners observed in the active grid pair"""
        if self.current_bundle[portA] and self.current_bundle[portB]:
            ids_A = self.current_bundle[portA]["ids"]
            ids_B = self.current_bundle[portB]["ids"]
            common_ids = np.intersect1d(ids_A, ids_B)
            common_ids = common_ids.tolist()

        else:
            common_ids = []

        return common_ids

    def stereo_calibrate(self):
        """Iterates across all camera pairs. Intrinsic parameters are pulled
        from camera and combined with obj and img points for each pair.
        """

        # stereocalibration_flags = cv2.CALIB_USE_INTRINSIC_GUESS
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.00001)

        for pair, inputs in self.stereo_inputs.items():

            camA = self.synchronizer.streams[pair[0]].camera
            camB = self.synchronizer.streams[pair[1]].camera

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
                camera_matrix_32,
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

            # TODO: update config outside of the stereocalibrator
            # self.session.config[f"stereo_{pair[0]}_{pair[1]}_rotation"] = rotation
            # self.session.config[f"stereo_{pair[0]}_{pair[1]}_translation"] = translation
            # self.session.config[f"stereo_{pair[0]}_{pair[1]}_RMSE"] = ret

            logging.info(
                f"For camera pair {pair}, rotation is \n{rotation}\n and translation is \n{translation}"
            )
            logging.info(f"RMSE of reprojection is {ret}")
        # self.session.update_config()

    def remove_full_pairs(self):

        for pair in self.uncalibrated_pairs:
            grid_count = len(self.stereo_inputs[pair]["common_board_loc"])

            if grid_count > self.grid_count_trigger:
                self.uncalibrated_pairs.remove(pair)

    def get_common_locs(self, port, common_ids):
        """Pull out objective location and image location of board corners for
        a port that are on the list of common ids"""

        ids = self.current_bundle[port]["ids"]
        img_loc = self.current_bundle[port]["img_loc"].squeeze().tolist()
        board_loc = self.current_bundle[port]["board_loc"].squeeze().tolist()

        common_img_loc = []
        common_board_loc = []

        for crnr_id, img, obj in zip(ids, img_loc, board_loc):
            if crnr_id in common_ids:
                common_board_loc.append(img)
                common_img_loc.append(obj)

        return common_img_loc, common_board_loc


if __name__ == "__main__":
    from src.calibration.corner_tracker import CornerTracker
    from src.session import Session

    logging.debug("Test live stereocalibration processing")

    repo = Path(__file__).parent.parent.parent
    config_path = Path(repo, "sessions", "default_session")
    session = Session(config_path)

    session.load_cameras()
    session.load_stream_tools()
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
        bundle = stereo_cal.current_bundle

        for port in bundle.keys():
            if bundle[port] is not None:
                cv2.imshow(str(port), bundle[port]["frame"])

        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
    logging.debug(pprint.pformat(stereo_cal.stereo_inputs))
