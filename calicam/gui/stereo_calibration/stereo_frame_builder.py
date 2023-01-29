import logging

LOG_FILE = "log\stereoframe_builder.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from pathlib import Path

import cv2
import numpy as np

from calicam.cameras.synchronizer import Synchronizer


class StereoFrameBuilder:
    def __init__(self, stereo_calibrator, single_frame_height=250):
        self.stereo_calibrator = stereo_calibrator
        self.single_frame_height = single_frame_height

        self.get_camera_rotation()

    def get_camera_rotation(self):
        # need to get rotation of each camera/port
        # not happy with how deep this reach is into the object hierarchy,
        # but doing rotation at the end avoids having to appy corrections for
        # drawing of corners during stereocalibration
        self.rotation_counts = {}
        for port, stream in self.stereo_calibrator.synchronizer.streams.items():
            self.rotation_counts[port] = stream.camera.rotation_count

    def set_current_synched_frames(self):
        self.stereo_calibrator.cal_frames_ready_q.get()  # impose wait until update
        self.current_synched_frames = self.stereo_calibrator.current_synched_frames

    def draw_common_corner_current(self, frameA, portA, frameB, portB):
        """Return unaltered frame if no corner information detected, otherwise
        return two frames with same corners drawn"""
        if self.current_synched_frames[portA] is None:
            logging.warn(f"Dropped frame at port {portA}")
            return frameA, frameB

        elif self.current_synched_frames[portB] is None:
            logging.warn(f"Dropped frame at port {portB}")
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
                    x = round(float(img_loc[0,0]))
                    y = round(float(img_loc[0,1]))

                    cv2.circle(frameA, (x, y), 5, (0, 0, 220), 3)

            for _id, img_loc in zip(ids_B, img_loc_B):
                if _id in common_ids:
                    x = round(float(img_loc[0,0]))
                    y = round(float(img_loc[0,1]))

                    cv2.circle(frameB, (x, y), 5, (0, 0, 220), 3)
            return frameA, frameB

    def draw_common_corner_history(self, frameA, portA, frameB, portB):

        pair = (portA, portB)
        img_loc_A = self.stereo_calibrator.stereo_inputs[pair]["img_loc_A"]
        img_loc_B = self.stereo_calibrator.stereo_inputs[pair]["img_loc_B"]

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
        logging.debug("resizing square")

        frame = cv2.flip(frame, 1)

        height = frame.shape[0]
        width = frame.shape[1]

        padded_size = max(height, width)

        height_pad = int((padded_size - height) / 2)
        width_pad = int((padded_size - width) / 2)
        pad_color = [0, 0, 0]

        logging.debug("about to pad border")
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
            logging.debug("plugging blank frame data")
            frame = np.zeros((edge, edge, 3), dtype=np.uint8)
        else:
            frame = self.current_synched_frames[port]["frame"]

        frame = frame.copy()
        return frame

    def hstack_frames(self, pair):
        """place paired frames side by side"""

        portA, portB = pair
        logging.debug("Horizontally stacking paired frames")
        frameA = self.get_frame_or_blank(portA)
        frameB = self.get_frame_or_blank(portB)

        frameA, frameB = self.draw_common_corner_history(frameA, portA, frameB, portB)
        frameA, frameB = self.draw_common_corner_current(frameA, portA, frameB, portB)

        frameA = self.resize_to_square(frameA)
        frameB = self.resize_to_square(frameB)

        frameA = self.apply_rotation(frameA, portA)
        frameB = self.apply_rotation(frameB, portB)

        hstacked_pair = np.hstack((frameA, frameB))

        return hstacked_pair

    def get_stereoframe_pairs(self):
        """Build a dictionary of paired frames to be broadcast to interface"""
        frame_pairs = {}
        for pair in self.stereo_calibrator.pairs:
            frame_pairs[pair] = self.hstack_frames(pair)
        return frame_pairs

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
    
def resize(image, new_height):
    # Get current dimensions
    (current_height, current_width) = image.shape[:2]

    # ratio to scale by
    ratio = new_height / float(current_height)
    
    # New dimensions
    dim = (int(current_width * ratio), new_height)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized


if __name__ == "__main__":
    from calicam.calibration.corner_tracker import CornerTracker
    from calicam.calibration.stereocalibrator import StereoCalibrator
    from calicam.session import Session

    logging.debug("Test live stereocalibration processing")

    repo = Path(__file__).parent.parent.parent.parent
    config_path = Path(repo, "sessions", "high_res_session")
    print(config_path)

    session = Session(config_path)
    session.load_cameras()
    session.load_streams()
    session.adjust_resolutions()
    # time.sleep(3)

    trackr = CornerTracker(session.charuco)

    logging.info("Creating Synchronizer")
    syncr = Synchronizer(session.streams, fps_target=4)
    logging.info("Creating Stereocalibrator")
    stereo_cal = StereoCalibrator(syncr, trackr)
    frame_builder = StereoFrameBuilder(stereo_cal)

    # while len(stereo_cal.uncalibrated_pairs) == 0:
    # time.sleep(.1)
    logging.info("Showing Stacked Frames")
    while len(stereo_cal.uncalibrated_pairs) > 0:
        # wait for newly processed frame to be available
        # frame_ready = frame_builder.stereo_calibrator.cal_frames_ready_q.get()
        frame_builder.set_current_synched_frames()

        for pair, frame in frame_builder.get_stereoframe_pairs().items():
            cv2.imshow(str(pair), frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
