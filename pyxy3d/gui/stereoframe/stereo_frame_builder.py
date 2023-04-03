import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from pathlib import Path

import cv2
import numpy as np

from pyxy3d.cameras.synchronizer import Synchronizer
from itertools import combinations
from queue import Queue
from threading import Event

COMMON_CORNER_TARGET = 5 # how many shared corners must be present to be recorded...

class StereoFrameBuilder:
    def __init__(self, synchronizer: Synchronizer, single_frame_height=250,board_count_target=50):
        self.synchronizer = synchronizer 
        self.single_frame_height = single_frame_height

        # used to determine sort order. Should this be in the stereo_tracker?
        self.board_count_target = board_count_target 
        self.common_corner_target = COMMON_CORNER_TARGET
        
        self.rotation_counts = {}
        for port, stream in self.synchronizer.streams.items():
            # override here while testing this out with pre-recorded video
            self.rotation_counts[port] = 0 # stream.camera.rotation_count

        self.pairs = self.get_pairs()

        self.new_sync_packet_notice = Queue()
        self.synchronizer.subscribe_to_notice(self.new_sync_packet_notice)
        self.store_points = Event()
    
        self.board_counts = {pair: 0 for pair in self.pairs} # TODO: part of future refactor to get way from stereotracker
        self.stereo_list = self.pairs.copy()
        self.stereo_history = {pair:{"img_loc_A":[], "img_loc_B":[]} for pair in self.pairs}
        self.store_points.clear()   # don't default to storing tracked points
        
        
        
    def get_pairs(self):
        pairs = [pair for pair in combinations(self.synchronizer.ports, 2)]
        sorted_ports = [
            (min(pair), max(pair)) for pair in pairs
        ]  # sort as in (b,a) --> (a,b)
        sorted_ports = sorted(
            sorted_ports
        )  # sort as in [(b,c), (a,b)] --> [(a,b), (b,c)]
        return sorted_ports

    def update_stereo_list(self):
        self.stereo_list = [
            key
            for key, value in self.board_counts.items()
            if value < self.board_count_target
        ]
        self.stereo_list = sorted(self.stereo_list, key=self.board_counts.get, reverse=True)

    def get_frame_or_blank(self, port):
        """Synchronization issues can lead to some frames being None among
        the synched frames, so plug that with a blank frame"""

        edge = self.single_frame_height
        frame_packet = self.current_sync_packet.frame_packets[port]
        if frame_packet is None:
            logger.debug("plugging blank frame data")
            frame = np.zeros((edge, edge, 3), dtype=np.uint8)
        else:
            frame = frame_packet.frame.copy()

        return frame

    def draw_common_corner_current(self, frameA, portA, frameB, portB):
        """
        Return unaltered frame if no corner information detected, otherwise
        record the corners that are shared between the two frames
        and return two frames with shared corners drawn
        """

        if self.current_sync_packet.frame_packets[portA] is None:
            logger.warning(f"Dropped frame at port {portA}")
            return frameA, frameB

        elif self.current_sync_packet.frame_packets[portB] is None:
            logger.warning(f"Dropped frame at port {portB}")
            return frameA, frameB

        frame_packet_A = self.current_sync_packet.frame_packets[portA]
        frame_packet_B = self.current_sync_packet.frame_packets[portB]

        if frame_packet_A.points is None or frame_packet_B.points is None:
            return frameA, frameB
        else:
            ids_A = frame_packet_A.points.point_id
            ids_B = frame_packet_B.points.point_id
            common_ids = np.intersect1d(ids_A, ids_B)

            img_loc_A = frame_packet_A.points.img_loc
            img_loc_B = frame_packet_B.points.img_loc

            # bit of an important side effect in here. Not best practice,
            # but the convenience is hard to pass up...
            # if there are enough corners in common, then store the corner locations
            # in the stereo history and update the board counts
            if len(common_ids) > self.common_corner_target and self.store_points.is_set():
                # logger.info("Storing common ids..")
                # logger.info("Stereo History:")
                # logger.info(self.stereo_history)
                self.stereo_history[(portA,portB)]['img_loc_A'].extend(img_loc_A.tolist())
                self.stereo_history[(portA,portB)]['img_loc_B'].extend(img_loc_B.tolist())
                self.board_counts[(portA,portB)]+=1
                
            for _id, img_loc in zip(ids_A, img_loc_A):
                if _id in common_ids:
                    x = round(float(img_loc[0]))
                    y = round(float(img_loc[1]))

                    cv2.circle(frameA, (x, y), 5, (0, 0, 220), 3)

            for _id, img_loc in zip(ids_B, img_loc_B):
                if _id in common_ids:
                    x = round(float(img_loc[0]))
                    y = round(float(img_loc[1]))

                    cv2.circle(frameB, (x, y), 5, (0, 0, 220), 3)
            return frameA, frameB

    def draw_common_corner_history(self, frameA, portA, frameB, portB):
        pair = (portA, portB)
        img_loc_A = self.stereo_history[pair]["img_loc_A"]
        img_loc_B = self.stereo_history[pair]["img_loc_B"]

        for coord_xy in img_loc_A:
            corner = (int(coord_xy[0]), int(coord_xy[1]))
            cv2.circle(frameA, corner, 2, (255, 165, 0), 2, 1)

        for coord_xy in img_loc_B:
            corner = (int(coord_xy[0]), int(coord_xy[1]))
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

    def hstack_frames(self, pair, board_count):
        """place paired frames side by side with an info box to the left"""

        portA, portB = pair
        logger.debug("Horizontally stacking paired frames")
        frameA = self.get_frame_or_blank(portA)
        frameB = self.get_frame_or_blank(portB)

        # this will be part of next round of refactor
        frameA, frameB = self.draw_common_corner_history(frameA, portA, frameB, portB)
        frameA, frameB = self.draw_common_corner_current(frameA, portA, frameB, portB)

        frameA = self.resize_to_square(frameA)
        frameB = self.resize_to_square(frameB)

        frameA = self.apply_rotation(frameA, portA)
        frameB = self.apply_rotation(frameB, portB)


        frameA = cv2.putText(frameA,
                             str(portA),
                            (int(frameA.shape[1]/2), int(self.single_frame_height / 4)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=2,
                        )

        frameB = cv2.putText(frameB,
                             str(portB),
                            (int(frameB.shape[1]/2), int(self.single_frame_height / 4)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=2,
                        )

        
        hstacked_pair = np.hstack(( frameA, frameB))
        
        hstacked_pair = cv2.putText(
            hstacked_pair,
            str(board_count),
            (self.single_frame_height-22,int(self.single_frame_height*4/5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0,0,255),
            thickness=2,
             
        )

        return hstacked_pair


    def get_completion_frame(self):
        height = int(self.single_frame_height)
        height = 1
        # because the label box to the left is half of the single frame width
        width = int(self.single_frame_height * 2.5)

        blank = np.zeros((height, width, 3), np.uint8)

        blank = cv2.putText(
            blank,
            "",
            (20, int(self.single_frame_height / 2)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 165, 0),
            thickness=1,
        )
        
        return blank
    
    def get_stereo_frame(self):
        """
        This glues together the stereopairs with summary blocks of the common board count
        """

        self.new_sync_packet_notice.get()
        self.current_sync_packet = self.synchronizer.current_sync_packet

        stereo_frame = None
        board_target_reached = False
        for pair in self.stereo_list:

            # figure out if you need to update the stereo frame list
            board_count = self.board_counts[pair]
            if board_count > self.board_count_target - 1:
                board_target_reached = True

            if stereo_frame is None:
                stereo_frame = self.hstack_frames(pair, board_count)
            else:
                stereo_frame = np.vstack(
                    [stereo_frame, self.hstack_frames(pair, board_count)]
                )

        if board_target_reached:
            self.update_stereo_list()

        if stereo_frame is None:
            stereo_frame = self.get_completion_frame()
        return stereo_frame


def resize(image, new_height):
    (current_height, current_width) = image.shape[:2]
    ratio = new_height / float(current_height)
    dim = (int(current_width * ratio), new_height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

if __name__ == "__main__":
    from pyxy3d.calibration.corner_tracker import CornerTracker
    from pyxy3d.recording.recorded_stream import RecordedStreamPool,RecordedStream
    from pyxy3d.recording.video_recorder import VideoRecorder
    from pyxy3d.session import Session
    from pyxy3d.calibration.charuco import Charuco
    
    from pyxy3d import __root__
    
    ports = [0, 1, 2, 3, 4]
    # ports = [1,2, 3]

    test_live = True
    record = True
    # test_live = False
    
    if test_live:

        session_directory = Path(__root__, "tests", "4_cameras_beginning")
        session = Session(session_directory)
        session.load_cameras()
        session.load_streams()
        logger.info("Creating Synchronizer")
        syncr = Synchronizer(session.streams, fps_target=3)
    else:
        recording_directory = Path(__root__, "tests", "mimic_anipose")
        charuco = Charuco(
            4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
        )
        recorded_stream_pool = RecordedStreamPool(ports, recording_directory, charuco=charuco)
        logger.info("Creating Synchronizer")
        syncr = Synchronizer(recorded_stream_pool.streams, fps_target=3)
        recorded_stream_pool.play_videos()

    frame_builder = StereoFrameBuilder(synchronizer=syncr, board_count_target=10)

    if record:
        video_path = Path(__root__, "tests", "please work")
        video_recorder = VideoRecorder(syncr)
        video_recorder.start_recording(video_path)

    
    while not syncr.stop_event.is_set():

        stereo_frame = frame_builder.get_stereo_frame()
        if stereo_frame is None:
            cv2.destroyAllWindows()
            break

        cv2.imshow("stereo frame", stereo_frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        
        if key == ord("s"): # as in `s`tore
            if frame_builder.store_points.is_set():
                frame_builder.store_points.clear()
            else: 
                frame_builder.store_points.set()
        
        if key == ord("u"):
            frame_builder.synchronizer.unsubscribe_to_streams() 
    # recorder.stop_recording()
    cv2.destroyAllWindows()
    
    if record:
        video_recorder.stop_recording()