import logging
logging.basicConfig(filename="stereocalibration.log", 
                    filemode = "w", 
                    level=logging.DEBUG)
                    # level=logging.INFO)

import pickle
import cv2
import time
from itertools import combinations
# from matplotlib.pyplot import grid
import numpy as np
import imutils
import pprint
from queue import Queue
from threading import Thread
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.session import Session
from src.calibration.synchronizer import Synchronizer


class StereoCalibrator:
    logging.info("Building Stereocalibrator...")
    def __init__(self, synchronizer):
        
        self.synchronizer = synchronizer
        self.session = synchronizer.session
        self.calibration_started = False        
        # when this many frames of conrners synched, move on to calibration
        self.grid_count_trigger = 15 
        self.wait_time = .5  # seconds between snapshots
        # board corners in common for a snapshot to be taken
        self.corner_threshold = 7

        self.stacked_frames = Queue() 

        self.ports = []
        logging.debug("Initializing ports...")
        for port, camera in self.session.camera.items():
            logging.debug(f"Appending port {port}...")
            self.ports.append(port)

        self.uncalibrated_pairs = [(i,j) for i,j in combinations(self.ports,2)]
        self.last_store_time = {pair: time.perf_counter() for pair in self.uncalibrated_pairs}        

        logging.debug(f"Processing pairs of uncalibrated pairs: {self.uncalibrated_pairs}")
        self.stereo_inputs = {pair:{"obj":[], "img_A": [], "img_B": []} for pair in self.uncalibrated_pairs}
        self.thread = Thread(target=self.push_bundled_frames, args=(), daemon=True)
        self.thread.start()

    
    def push_bundled_frames(self):
        logging.debug(f"Currently {len(self.uncalibrated_pairs)} uncalibrated pairs ")
        while len(self.uncalibrated_pairs) >0:
            frame_bundle = self.synchronizer.synced_frames_q.get()
            # frame_bundle = self.synchronizer.synced_frames_q.get()
            self.process_frame_bundle(frame_bundle)
            logging.debug("About to push bundled frame to stack")
            self.stacked_frames.put(self.superframe(single_frame_height=250))
            # cv2.imshow("Stereocalibration", stacked_pairs)

            self.remove_full_pairs()

            if len(self.uncalibrated_pairs) == 0:
                self.stereo_calibrate()
                # self.calibration = Thread(target=self.stereo_calibrate(), args=(), daemon=True)
                # self.calibration.start()
                # self.calibration_started = True
                # while True:
                    # time.sleep(.5)
                    # self.stacked_frames.put(np.array([]))


    def stereo_calibrate(self):
        """ Iterates across all camera pairs. Intrinsic parameters are pulled
        from camera and combined with obj and img points for each pair. 
        """

        # stereocalibration_flags = cv2.CALIB_USE_INTRINSIC_GUESS
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.00001) 

        for pair, inputs in self.stereo_inputs.items():

            camA = self.synchronizer.session.camera[pair[0]]
            camB = self.synchronizer.session.camera[pair[1]]

            obj = self.stereo_inputs[pair]["obj"]
            img_A = self.stereo_inputs[pair]["img_A"]
            img_B = self.stereo_inputs[pair]["img_B"]

            # convert to list of vectors for OpenCV function
            obj = [np.array(x, dtype=np.float32) for x in obj]
            img_A = [np.array(x, dtype=np.float32) for x in img_A]
            img_B =[np.array(x, dtype=np.float32) for x in img_B]

            (ret, 
            camera_matrix_1, 
            distortion_1, 
            camera_matrix_32, 
            distortion_2,
            rotation, 
            translation, 
            essential,
            fundamental) = cv2.stereoCalibrate(obj,
                                               img_A,
                                               img_B,
                                               camA.camera_matrix,  
                                               camA.distortion, 
                                               camB.camera_matrix,  
                                               camB.distortion, 
                                               imageSize = None, # (400, 400), # (width, height)....my recollection is that these did not matter. from OpenCV: "Size of the image used only to initialize the camera intrinsic matrices."
                                               criteria = criteria, 
                                               flags = stereocalibration_flags,
                                               )

            self.session.config[f"stereo_{pair[0]}_{pair[1]}_rotation"] = rotation
            self.session.config[f"stereo_{pair[0]}_{pair[1]}_translation"] = translation 
            self.session.config[f"stereo_{pair[0]}_{pair[1]}_RMSE"] = ret 

            logging.info(f"For camera pair {pair}, rotation is \n{rotation}\n and translation is \n{translation}")
            logging.info(f"RMSE of reprojection is {ret}")
        self.session.update_config()
        
    def remove_full_pairs(self):
        
        for pair in self.uncalibrated_pairs:
            grid_count = len(self.stereo_inputs[pair]["obj"])

            if grid_count > self.grid_count_trigger:
                self.uncalibrated_pairs.remove(pair)


    def process_frame_bundle(self, frame_bundle,  ):
        logging.debug("About to process current frame bundle")        
        self.frame_bundle = frame_bundle

        for pair in self.uncalibrated_pairs:
            portA = pair[0]
            portB = pair[1]

            self.set_common_ids(portA, portB)

            enough_corners = len(self.common_ids) > self.corner_threshold
            enough_time = time.perf_counter() - self.last_store_time[pair] > self.wait_time

            if enough_corners and enough_time:
                obj, img_A = self.get_obj_img_points(portA)
                _, img_B = self.get_obj_img_points(portB)
               
                self.stereo_inputs[pair]["obj"].append(obj)
                self.stereo_inputs[pair]["img_A"].append(img_A)
                self.stereo_inputs[pair]["img_B"].append(img_B)
                self.last_store_time[pair] = time.perf_counter()

    def superframe(self, single_frame_height=250):
        """Combine all current bundled frames into a grid. Paired frames are 
        next to each other horizontally and all pairs are vertically stacked"""
        logging.debug("assembling superframe")

        self.single_frame_height = single_frame_height
        stacked_bundle = np.array([])

        for pair in self.uncalibrated_pairs:

            portA = pair[0]
            portB = pair[1]        

            hstacked_pair = self.hstack_frames(portA, portB)

            if stacked_bundle.any():
                stacked_bundle = np.vstack((stacked_bundle, hstacked_pair))
            else:
                stacked_bundle = hstacked_pair

        return stacked_bundle

    def resize_to_square(self, frame):
        """ To make sure that frames align well, scale them all to thumbnails
        squares with black borders. If there is a dropped frame, make the image
        blank."""
        logging.debug("resizing square")
        edge = self.single_frame_height # square edge length

        frame = cv2.flip(frame,1)

        height = frame.shape[0]    
        width = frame.shape[1]

        padded_size = max(height,width)

        height_pad = int((padded_size-height)/2)
        width_pad = int((padded_size-width)/2)
        pad_color = [0,0,0]

        logging.debug("about to pad border")
        frame = cv2.copyMakeBorder(frame, 
                                    height_pad,height_pad, 
                                    width_pad,width_pad, 
                                    cv2.BORDER_CONSTANT,value = pad_color)

        frame = imutils.resize(frame, height=edge)
        return frame    

    def set_common_ids(self, portA, portB):
        """Intersection of grid corners observed in the active grid pair"""
        if self.frame_bundle[portA] and self.frame_bundle[portB]:
            corner_idsA = self.frame_bundle[portA]["corner_ids"]
            corner_idsB = self.frame_bundle[portB]["corner_ids"]
            common_ids = np.intersect1d(corner_idsA,corner_idsB)
            common_ids = common_ids.tolist()

            self.common_ids =  common_ids
        else:
            self.common_ids = []

    def get_obj_img_points(self, port):
        """Pull out objective location and image location of board corners for 
        a port that are on the list of common ids"""

        corner_ids = self.frame_bundle[port]["corner_ids"]
        frame_corners = self.frame_bundle[port]["frame_corners"].squeeze().tolist()
        board_FOR_corners = self.frame_bundle[port]["board_FOR_corners"].squeeze().tolist()   

        obj_points = []
        img_points = []

        for crnr_id, img, obj in zip(corner_ids, frame_corners, board_FOR_corners):
            if crnr_id in self.common_ids:
                img_points.append(img)
                obj_points.append(obj)

        return obj_points, img_points 

    def draw_stored_corners(self, frameA, portA, frameB, portB):

        pair = (portA, portB)
        img_A = self.stereo_inputs[pair]['img_A']
        img_B = self.stereo_inputs[pair]['img_B']


        for cornerset in img_A:
            for corner in cornerset:
                corner = (int(corner[0]), int(corner[1]))
                cv2.circle(frameA, corner,2,(255,165,0), 2,1)

        for cornerset in img_B:
            for corner in cornerset:
                corner = (int(corner[0]), int(corner[1]))
                cv2.circle(frameB, corner,2,(255,165,0), 2,1)


        return frameA, frameB
            
    def frame_or_blank(self, port):
        logging.debug("plugging blank frame data")

        edge = self.single_frame_height
        bundle = self.frame_bundle[port]
        if bundle is None:
            frame = np.zeros((edge, edge, 3), dtype=np.uint8)
        else:
            frame = self.frame_bundle[port]["frame"]

        frame = frame.copy()
        return frame


    def hstack_frames(self, portA, portB):
        """place paired frames side by side"""
        logging.debug("Horizontally stacking paired frames")
        frameA = self.frame_or_blank(portA)
        frameB = self.frame_or_blank(portB)

        frameA, frameB = self.draw_stored_corners(frameA, portA, frameB, portB)

        
        frameA = cv2.flip(frameA,1)
        cv2.putText(frameA, f"Port: {portA}", (30,50), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,200), 3)
        frameA = cv2.flip(frameA,1)

        frameB = cv2.flip(frameB,1)
        cv2.putText(frameB, f"Port: {portB}", (30,50), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,200), 3)
        frameB = cv2.flip(frameB,1)

        frameA = self.resize_to_square(frameA)
        frameB = self.resize_to_square(frameB)

        # cv2.imshow(str(portA), frameA)
        # cv2.imshow(str(portB), frameB)

        stacked_pair = np.hstack((frameA, frameB))

        return stacked_pair


if __name__ == "__main__":


    logging.debug("Test live stereocalibration processing")
    session = Session("test_session")
    session.load_cameras()
    session.load_streams()
    session.adjust_resolutions()
    # time.sleep(3)
    logging.info("Creating Synchronizer")
    syncr = Synchronizer(session, fps_target=6)
    logging.info("Creating Stereocalibrator")
    stereo_cal = StereoCalibrator(syncr)

    # while len(stereo_cal.uncalibrated_pairs) == 0:
        # time.sleep(.1)
    logging.info("Showing Stacked Frames")
    while len(stereo_cal.uncalibrated_pairs) > 0:
            
        frame = stereo_cal.stacked_frames.get()
        print(frame.dtype)
        print(frame.shape)
        if frame.shape == (1,):
            logging.info("Beginning to calibrate")
            cv2.destroyAllWindows()
        cv2.imshow("Stereocalibration", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
    


    cv2.destroyAllWindows()
    logging.debug(pprint.pformat(stereo_cal.stereo_inputs))