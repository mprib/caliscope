import calicam.logger

logger = calicam.logger.get(__name__)

from pathlib import Path

import cv2
import numpy as np

from calicam.cameras.synchronizer import Synchronizer


class StereoPairFrameBuilder:
    def __init__(self, stereo_tracker, single_frame_height=250):
        self.stereo_tracker = stereo_tracker
        self.single_frame_height = single_frame_height
        self.board_count_target = 50    # used to determine sort order. 

        self.rotation_counts = {}
        for port, stream in self.stereo_tracker.synchronizer.streams.items():
            self.rotation_counts[port] = stream.camera.rotation_count

        self.omni_frame_order = self.stereo_tracker.pairs.copy()
        self.omni_frame_order.sort()
        
        self.board_counts = {pair:0 for pair in self.omni_frame_order}

    def get_new_raw_frames(self):
        self.stereo_tracker.cal_frames_ready_q.get()  # impose wait until update
        self.current_synched_frames = self.stereo_tracker.current_synched_frames

        # update the board_counts here
        
        
        
    def draw_common_corner_current(self, frameA, portA, frameB, portB):
        """Return unaltered frame if no corner information detected, otherwise
        return two frames with same corners drawn"""
        if self.current_synched_frames[portA] is None:
            logger.warn(f"Dropped frame at port {portA}")
            return frameA, frameB

        elif self.current_synched_frames[portB] is None:
            logger.warn(f"Dropped frame at port {portB}")
            return frameA, frameB

        elif (
            "ids" not in self.current_synched_frames[portA]
            or "ids" not in self.current_synched_frames[portB]
        ):
            return frameA, frameB
        else:
            ids_A = self.current_synched_frames[portA]["ids"]
            ids_B = self.current_synched_frames[portB]["ids"]
            common_ids = np.intersect1d(ids_A, ids_B)

            img_loc_A = self.current_synched_frames[portA]["img_loc"]
            img_loc_B = self.current_synched_frames[portB]["img_loc"]

            for _id, img_loc in zip(ids_A, img_loc_A):
                if _id in common_ids:
                    x = round(float(img_loc[0, 0]))
                    y = round(float(img_loc[0, 1]))

                    cv2.circle(frameA, (x, y), 5, (0, 0, 220), 3)

            for _id, img_loc in zip(ids_B, img_loc_B):
                if _id in common_ids:
                    x = round(float(img_loc[0, 0]))
                    y = round(float(img_loc[0, 1]))

                    cv2.circle(frameB, (x, y), 5, (0, 0, 220), 3)
            return frameA, frameB

    def draw_common_corner_history(self, frameA, portA, frameB, portB):

        pair = (portA, portB)
        img_loc_A = self.stereo_tracker.stereo_inputs[pair]["img_loc_A"]
        img_loc_B = self.stereo_tracker.stereo_inputs[pair]["img_loc_B"]

        for cornerset in img_loc_A:
            for corner in cornerset:
                corner = (int(corner[0][0]), int(corner[0][1]))
                cv2.circle(frameA, corner, 2, (255, 165, 0), 2, 1)

        for cornerset in img_loc_B:
            for corner in cornerset:
                corner = (int(corner[0][0]), int(corner[0][1]))
                cv2.circle(frameB, corner, 2, (255, 165, 0), 2, 1)

        return frameA, frameB

    def resize_to_square(self, frame):
        """To make sure that frames align well, scale them all to thumbnails
        squares with black borders."""
        logger.debug("resizing square")

        frame = cv2.flip(frame, 1)

        height = frame.shape[0]
        width = frame.shape[1]

        padded_size = max(height, width)

        height_pad = int((padded_size - height) / 2)
        width_pad = int((padded_size - width) / 2)
        pad_color = [0, 0, 0]

        logger.debug("about to pad border")
        frame = cv2.copyMakeBorder(
            frame,
            height_pad,
            height_pad,
            width_pad,
            width_pad,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )

        frame = resize(frame, new_height=self.single_frame_height)
        return frame

    def get_frame_or_blank(self, port):
        """Synchronization issues can lead to some frames being None among
        the synched frames, so plug that with a blank frame"""

        edge = self.single_frame_height
        synched_frames = self.current_synched_frames[port]
        if synched_frames is None:
            logger.debug("plugging blank frame data")
            frame = np.zeros((edge, edge, 3), dtype=np.uint8)
        else:
            frame = self.current_synched_frames[port]["frame"]

        frame = frame.copy()
        return frame

    def hstack_frames(self, pair,  board_count):
        """place paired frames side by side"""

        portA, portB = pair
        logger.debug("Horizontally stacking paired frames")
        frameA = self.get_frame_or_blank(portA)
        frameB = self.get_frame_or_blank(portB)

        frameA, frameB = self.draw_common_corner_history(frameA, portA, frameB, portB)
        frameA, frameB = self.draw_common_corner_current(frameA, portA, frameB, portB)

        frameA = self.resize_to_square(frameA)
        frameB = self.resize_to_square(frameB)

        frameA = self.apply_rotation(frameA, portA)
        frameB = self.apply_rotation(frameB, portB)

        label_display = np.zeros(
            (self.single_frame_height, int(self.single_frame_height / 2), 3), np.uint8
        )
        label_display = cv2.putText(
            label_display,
            f"{pair[0]} | {pair[1]}",
            (10, int(self.single_frame_height/3)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 165, 0),
            thickness=1,
        )
        
        label_display = cv2.putText(
            label_display,
            str(board_count),
            (10, int(self.single_frame_height*(2/3))),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 165, 0),
            thickness=1,
        )

        hstacked_pair = np.hstack((label_display, frameA, frameB))

        return hstacked_pair

    def get_stereoframes(self):
        """Build a dictionary of paired frames to be broadcast to interface"""
        stereo_frames = {}
        for pair in self.stereo_tracker.pairs:
            stereo_frames[pair] = self.hstack_frames(pair)
        return stereo_frames

    def get_pair_board_counts(self):
        board_counts = {}
        for pair in self.omni_frame_order:
            corner_history = self.stereo_tracker.stereo_inputs[pair]
            board_counts[pair] = len(corner_history["common_board_loc"])

        return board_counts

    def apply_rotation(self, frame, port):
        rotation_count = self.rotation_counts[port]
        if rotation_count == 0:
            pass
        elif rotation_count in [1, -3]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_count in [2, -2]:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_count in [-1, 3]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return frame

    def get_omni_frame(self):
        """
        This glues together the stereopairs with summary blocks of the common board count
        """

        pair_board_counts = self.get_pair_board_counts()

        omni_frame = None
        for pair in self.omni_frame_order:
            board_count = pair_board_counts[pair]
            if omni_frame is None:
                omni_frame = self.hstack_frames(pair, board_count)
            else:
                omni_frame = np.vstack([omni_frame, self.hstack_frames(pair, board_count)])

        return omni_frame
    
    def reorder_omni_frame(self):

        pass


def resize(image, new_height):
    (current_height, current_width) = image.shape[:2]
    ratio = new_height / float(current_height)
    dim = (int(current_width * ratio), new_height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


if __name__ == "__main__":
    from calicam.calibration.corner_tracker import CornerTracker
    from calicam.calibration.stereocalibrator import StereoTracker
    from calicam.session import Session

    logger.debug("Test live stereocalibration processing")

    repo = Path(str(Path(__file__)).split("calicam")[0], "calicam")
    # config_path = Path(repo, "sessions", "high_res_session")
    config_path = Path(repo, "sessions", "5_cameras")
    print(config_path)

    session = Session(config_path)
    session.load_cameras()
    session.load_streams()
    session.adjust_resolutions()
    # time.sleep(3)

    trackr = CornerTracker(session.charuco)

    logger.info("Creating Synchronizer")
    syncr = Synchronizer(session.streams, fps_target=2)
    logger.info("Creating Stereocalibrator")
    stereo_cal = StereoTracker(syncr, trackr)
    frame_builder = StereoPairFrameBuilder(stereo_cal)

    # while len(stereo_cal.uncalibrated_pairs) == 0:
    # time.sleep(.1)
    logger.info("Showing Paired Frames")
    while len(stereo_cal.pairs) > 0:
        # wait for newly processed frame to be available
        # frame_ready = frame_builder.stereo_calibrator.cal_frames_ready_q.get()
        frame_builder.get_new_raw_frames()
        board_counts = frame_builder.get_pair_board_counts()

        omni_frame = frame_builder.get_omni_frame()

        cv2.imshow("omni frame", omni_frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
