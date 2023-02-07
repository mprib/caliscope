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

    def __init__(self, synchronizer, corner_tracker):

        self.corner_tracker = corner_tracker
        self.synchronizer = synchronizer

        self.corner_threshold = 7  # board corners in common for capture
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

        while not self.stop_event.set():
            self.synched_frames_available_q.get()

            # may get hung up on get, so additional item put on queue
            if self.stop_event.set():
                break

            self.current_synched_frames = self.synchronizer.current_synched_frames

            self.add_corner_data()
            logger.debug(
                "Begin determination of shared corners within current frame pairs"
            )
            for pair in self.pairs:
                self.store_stereo_data(pair)

            self.cal_frames_ready_q.put("frames ready")

        logger.info(
            "Stereocalibration synched frames harvester successfully shut-down..."
        )

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

                logger.debug(f"At port {port} the following corners are located {ids}")

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
        logger.info(f"About to stereocalibrate pair {pair}")

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

        if pair in self.pairs:
            logger.info(f"Removing pair {pair}")
            self.pairs.remove(pair)
        else:
            logger.warning(f"Attempted to remove pair {pair} but it was not present")

        logger.info(
            f"For camera pair {pair}, rotation is \n{rotation}\n and translation is \n{translation}"
        )
        logger.info(f"RMSE of reprojection is {ret}")

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


if __name__ == "__main__":
    import pprint

    from calicam.calibration.corner_tracker import CornerTracker
    from calicam.session import Session

    logger.debug("Test live stereocalibration processing")

    repo = Path(str(Path(__file__)).split("calicam")[0], "calicam")
    session_path = Path(repo, "sessions", "high_res_session")
    session = Session(session_path)

    session.load_cameras()
    session.load_streams()
    session.adjust_resolutions()
    # time.sleep(3)

    trackr = CornerTracker(session.charuco)

    logger.info("Creating Synchronizer")
    syncr = Synchronizer(session.streams, fps_target=6)
    logger.info("Creating Stereocalibrator")
    stereo_cal = StereoTracker(syncr, trackr)

    # while len(stereo_cal.uncalibrated_pairs) == 0:
    # time.sleep(.1)
    logger.info("Showing Stacked Frames")
    while len(stereo_cal.pairs) > 0:

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
    logger.debug(pprint.pformat(stereo_cal.stereo_inputs))
