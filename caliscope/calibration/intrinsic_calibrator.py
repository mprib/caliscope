import caliscope.logger
from queue import Queue
from threading import Thread, Event
import random
import cv2

from caliscope.packets import FramePacket
from caliscope.recording.recorded_stream import RecordedStream
from caliscope.cameras.camera_array import CameraData

logger = caliscope.logger.get(__name__)


class IntrinsicCalibrator:
    """
    Takes a recorded stream and determines a CameraData object from it
    Stream needs to have a charuco tracker assigned to it
    """

    def __init__(self, camera_data: CameraData, stream: RecordedStream):
        self.camera = camera_data  # pointer needed to update params
        self.stream = stream
        self.initialize_point_history()

        self.frame_packet_q = Queue()
        self.stream.subscribe(self.frame_packet_q)

        # The following group of parameters relate to the autopopulation of the calibrator
        self.grid_history_q = Queue()  # for passing ids, img_loc used in calibration 
        self.auto_store_data = Event()
        self.auto_store_data.clear()
        self.auto_pop_frame_wait = 0  # how many frames will you hold ofF checking if the board can be added to the calibration pile.
        self.target_grid_count = 0

        self.harvest_frames()
         
    def harvest_frames(self):
        self.stop_event = Event()
        self.stop_event.clear()
        
        def harvest_worker():
            while True:
                frame_packet = self.frame_packet_q.get()
                if self.stop_event.is_set():
                    break

                self.add_frame_packet(frame_packet)

            logger.info(f"Harvest frames successfully ended in calibrator for port {self.stream.port}")
        self.harvest_thread = Thread(target=harvest_worker, args=[], daemon=True)
        self.harvest_thread.start()

    def stop(self):
        logger.info("Beginning to stop intrinsic calibrator")
        self.stop_event.set()
        self.stream.unsubscribe(self.frame_packet_q)
        self.frame_packet_q.put(-1)

    @property
    def grid_count(self):
        """How many sets of corners have been collected up to this point"""
        return len(self.calibration_frame_indices)

    @property
    def image_size(self):
        image_size = list(self.camera.size)
        image_size.reverse()  # for some reason...
        image_size.append(3)

        return image_size

    def initialize_point_history(self):

        # list of frame_indices that will go into the calibration
        self.calibration_frame_indices = []

        # dictionaries here are indexed by frame_index
        self.all_ids = {}
        self.all_img_loc = {}
        self.all_obj_loc = {}

    def add_frame_packet(self, frame_packet: FramePacket):
        """
        Point data from frame packet is stored, indexed by the frame index
        """
        index = frame_packet.frame_index

        if index != -1:  # indicates end of stream
            self.all_ids[index] = frame_packet.points.point_id
            self.all_img_loc[index] = frame_packet.points.img_loc
            self.all_obj_loc[index] = frame_packet.points.obj_loc

            self.active_frame_index = index
        
            # when auto store data is set, the stream should be pushing out all
            # frames consecutively from the beginning
            if self.auto_store_data.is_set():
                point_id = frame_packet.points.point_id
                
                if point_id.size == 0:
                    corner_count = 0
                else:
                    corner_count = frame_packet.points.point_id.shape[0]
                logger.debug(f"Corner count is {corner_count} and frame wait is {self.auto_pop_frame_wait}")
                if self.auto_pop_frame_wait == 0 and corner_count >= self.threshold_corner_count:
                    # add frame to calibration data and reset the wait time
                    self.add_calibration_frame_index(index)
                    self.auto_pop_frame_wait = self.wait_between 
                else:
                    # count down to the next frame to consider autopopulating
                    self.auto_pop_frame_wait = max(self.auto_pop_frame_wait-1,0)       
                
                logger.debug(f"Current index is {index}")
                if index == self.stream.last_frame_index:
                # end of stream, so stop auto pop and backfill to hit grid target
                    logger.info("End of autopop detected...")
                    self.auto_store_data.clear()
                    self.backfill_calibration_frames()


    def backfill_calibration_frames(self):
        logger.info(f"Initiating backfill of frames to hit target grid count of {self.target_grid_count}...currently at {self.grid_count}")
        actual_grid_count = len(self.calibration_frame_indices)
        # build new frame list
        new_potential_frames = []
        for frame_index, ids in self.all_ids.items():
            if frame_index not in self.calibration_frame_indices:
                if len(ids) > 6: # believe this may be a requirement of the calibration algorithm
                    new_potential_frames.append(frame_index)
            
        sample_size = self.target_grid_count-actual_grid_count
        sample_size = min(sample_size, len(new_potential_frames))
        sample_size = max(sample_size,0)

        random_frames = random.sample(new_potential_frames,sample_size)
        for frame in random_frames:
            self.add_calibration_frame_index(frame)
        
    def add_calibration_frame_index(self, frame_index: int):
        """
        A "side effect" of this method is that the corner id and img_loc
        data is placed on a q. This q is consumed by the frame_emitter
        which will update the grid capture history based with those grids

        This allows the GUI element (frame emitter) to stay in sync with the 
        calibrator. 
        """
        logger.info(f"Adding frame data to calibration inputs for frame index {frame_index}")
        self.calibration_frame_indices.append(frame_index)
        
        # Backchannel communication to frame_emitter to keep things aligned
        ids = self.all_ids[frame_index]
        img_loc = self.all_img_loc[frame_index]
        self.grid_history_q.put((ids,img_loc))
        

    def clear_calibration_data(self):
        logger.info("Clearing calibration data..")
        self.calibration_frame_indices = []
        self.set_calibration_inputs()
        
    def initiate_auto_pop(self, wait_between,threshold_corner_count, target_grid_count):
        """
        This will enable actions within self.add_frame_packet
        
        Now when frame_packets are read in from the stream, the
        
        """
        logger.info(f"Initiating autopopulation of corner data in port {self.camera.port}")
        self.clear_calibration_data()
        self.wait_between = wait_between
        self.threshold_corner_count = threshold_corner_count        
        self.target_grid_count = target_grid_count
        self.initialize_point_history()
        self.auto_store_data.set()

    def set_calibration_inputs(self):
        """
        The data that will ultimately go into the calibration is determined by the calibration
        frame indices. This is where the calibration data is populated based on which frames
        have been flagged (either by the user via self.add_calibration_frame_index or by autopopulated
        within self.add_frame_packet when autopop is enabled.)
        """
        self.calibration_point_ids = []
        self.calibration_img_loc = []
        self.calibration_obj_loc = []
        logger.info(f"Blank calibration inputs initialized at port {self.camera.port }")
        for index in self.calibration_frame_indices:
            id_count = len(self.all_ids[index])
            if id_count > 3:  # I believe this is a requirement of opencv
                self.calibration_point_ids.append(self.all_ids[index])
                self.calibration_img_loc.append(self.all_img_loc[index])
                self.calibration_obj_loc.append(self.all_obj_loc[index])
            else:
                logger.info(f"Note that empty data stored in frame index {index}. This is not being used in the calibration")
        logger.info(f"Total size of inputs is {len(self.calibration_point_ids)}")

    def calibrate_camera(self):
        """
        Use the recorded image corner positions along with the objective
        corner positions based on the board definition to calculated
        the camera matrix and distortion parameters
        """
        self.set_calibration_inputs()

        logger.info(f"Calibrating camera {self.camera.port}....")

        width = self.stream.size[0]
        height = self.stream.size[1]

        self.error, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.calibration_obj_loc,
            self.calibration_img_loc,
            (width, height),
            None,
            None,
        )

        # fix extra dimension in return value of cv2.calibrateCamera
        self.dist = self.dist[0]

        self.update_camera()
        self.is_calibrated = True

        logger.info(f"Error: {self.error}")
        logger.info(f"Camera Matrix: {self.mtx}")
        logger.info(f"Distortion: {self.dist}")
        logger.info(f"Grid Count: {self.grid_count}")

    def update_camera(self):
        logger.info(f"Setting calibration params on camera {self.camera.port}")

        # ret is RMSE of reprojection
        self.camera.error = round(self.error, 3)
        self.camera.matrix = self.mtx
        self.camera.distortions = self.dist
        self.camera.grid_count = self.grid_count
