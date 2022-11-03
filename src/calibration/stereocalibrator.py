import pickle
from re import I
import cv2
import logging
import time
from itertools import combinations
from matplotlib.pyplot import grid
import numpy as np
import imutils
import pprint
from threading import Thread
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.session import Session
from src.calibration.synchronizer import Synchronizer

logging.basicConfig(filename="stereocalibration.log", 
                    filemode = "w", 
                    # level=logging.INFO)
                    level=logging.DEBUG)



class StereoCalibrator:

    def __init__(self, syncronizer):
        self.syncronizer = syncronizer

        # self.ports = 
        # self.uncalibrated_pairs = [(i,j) for i,j in combinations(self.syncronizer.ports,2)]
        self.uncalibrated_pairs = [(0,1)]
        self.stereo_inputs = {pair:{"obj":[], "img_A": [], "img_B": []} for pair in self.uncalibrated_pairs}
        self.last_store_time = time.perf_counter()        
        self.thread = Thread(target=self.show_bundled_frames, args=(), daemon=True)
        self.thread.start()

    def show_bundled_frames(self):
        while len(self.uncalibrated_pairs) >0:
            frame_bundle = self.syncronizer.synced_frames_q.get()
            self.process_frame_bundle(frame_bundle)
            
            stacked_pairs = stereo_cal.superframe(single_frame_height=250)
            cv2.imshow("Stereocalibration", stacked_pairs)

            self.remove_full_pairs()

            if len(self.uncalibrated_pairs) == 0:
                self.stereo_calibrate()
                # break

            key = cv2.waitKey(1)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break

    def stereo_calibrate(self):
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        
        for pair, inputs in self.stereo_inputs.items():
            camA = self.syncronizer.session.camera[pair[0]]
            camB = self.syncronizer.session.camera[pair[1]]

            obj = self.stereo_inputs[pair]["obj"]
            img_A = self.stereo_inputs[pair]["img_A"]
            img_B = self.stereo_inputs[pair]["img_B"]

            # convert to vectors
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
                                               (400, 400), # (width, height)....my recollection is that these did not matter. from OpenCV: "Size of the image used only to initialize the camera intrinsic matrices."
                                            #    criteria = criteria, 
                                               flags = stereocalibration_flags)

            print(f"For camera pair {pair}, rotation is {rotation} and translation is {translation}")


    def remove_full_pairs(self):

        for pair in self.uncalibrated_pairs:

            grid_count = len(self.stereo_inputs[pair]["obj"])

            if grid_count > 5:
                self.uncalibrated_pairs.remove(pair)
                # img_A = 


    def process_frame_bundle(self, frame_bundle, time_threshold=0.5, corner_threshold=7):
        self.frame_bundle = frame_bundle
        # self.corner_threshold = corner_threshold

        for pair in self.uncalibrated_pairs:
            portA = pair[0]
            portB = pair[1]
            self.set_common_ids(portA, portB)
            enough_corners = len(self.common_ids) > corner_threshold
            enough_time = time.perf_counter() - self.last_store_time > time_threshold

            if enough_corners and enough_time:
                obj, img_A = self.get_obj_img_points(portA)
                _, img_B = self.get_obj_img_points(portB)
                logging.debug(f"Common IDs for ports: {pair}")
                logging.debug(f"{self.common_ids}")
                logging.debug(f"Image points for port A: {img_A}")
                logging.debug(f"Image points for port B: {img_B}")
                
             #    if len(self.common_ids) > self.corner_threshold:
                self.stereo_inputs[pair]["obj"].append(obj)
                self.stereo_inputs[pair]["img_A"].append(img_A)
                self.stereo_inputs[pair]["img_B"].append(img_B)
                self.last_store_time = time.perf_counter()

    def superframe(self, single_frame_height=250):

        self.single_frame_height = single_frame_height
        stacked_bundle = np.array([])

        for pair in self.uncalibrated_pairs:

            portA = pair[0]
            portB = pair[1]        

            # create stacked frames to simplify video output
            hstacked_pair = self.hstack_frames(portA, portB)

            if stacked_bundle.any():
                stacked_bundle = np.vstack((stacked_bundle, hstacked_pair))
            else:
                stacked_bundle = hstacked_pair

        return stacked_bundle

    def resize_to_square(self, frame):
        """ returns a square image with black borders to round it out. If the 
        data is none, then makes the image completely blank"""

        edge = self.single_frame_height

        frame = cv2.flip(frame,1)

        height = frame.shape[0]    
        width = frame.shape[1]

        padded_size = max(height,width)

        height_pad = int((padded_size-height)/2)
        width_pad = int((padded_size-width)/2)
        pad_color = [0,0,0]

        frame = cv2.copyMakeBorder(frame, 
                                    height_pad,height_pad, 
                                    width_pad,width_pad, 
                                    cv2.BORDER_CONSTANT,value = pad_color)

        frame = imutils.resize(frame, height=edge)

        return frame    

    def set_common_ids(self, portA, portB):

        if self.frame_bundle[portA] and self.frame_bundle[portB]:
            corner_idsA = self.frame_bundle[portA]["corner_ids"]
            corner_idsB = self.frame_bundle[portB]["corner_ids"]
            common_ids = np.intersect1d(corner_idsA,corner_idsB)
            common_ids = common_ids.tolist()

            self.common_ids =  common_ids
        else:
            self.common_ids = []

    def get_obj_img_points(self, port):

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
        # else:
            # return [[]],[[]]

    def mark_stored_corners(self, frameA, portA, frameB, portB):
        pair = (portA, portB)
        img_A = self.stereo_inputs[pair]['img_A']
        img_B = self.stereo_inputs[pair]['img_B']

        # logging.debug(f"Marking Corners for pair: {pair}")
        # logging.debug(f

        for cornerset in img_A:
            for corner in cornerset:
                corner = (int(corner[0]), int(corner[1]))
                cv2.circle(frameA, corner,2,(255,165,0), 2,1)

        for cornerset in img_B:
            for corner in cornerset:
                corner = (int(corner[0]), int(corner[1]))
                cv2.circle(frameB, corner,2,(255,165,0), 2,1)

        # print(img_A, img_B)

        return frameA, frameB
            
    def frame_or_blank(self, port):

        edge = self.single_frame_height
        bundle = self.frame_bundle[port]
        if bundle is None:
            frame = np.zeros((edge, edge, 3), dtype=np.uint8)
        else:
            frame = self.frame_bundle[port]["frame"]

        frame = frame.copy()
        return frame


    def hstack_frames(self, portA, portB):

        frameA = self.frame_or_blank(portA)
        frameB = self.frame_or_blank(portB)

        frameA, frameB = self.mark_stored_corners(frameA, portA, frameB, portB)
        # frameB = self.mark_stored_corners(frameB, (portA, portB))

        frameA = cv2.flip(frameA,1)
        cv2.putText(frameA, f"Port: {portA}", (30,50), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,200), 3)
        frameA = cv2.flip(frameA,1)

        frameB = cv2.flip(frameB,1)
        cv2.putText(frameB, f"Port: {portB}", (30,50), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,200), 3)
        frameB = cv2.flip(frameB,1)

        frameA = self.resize_to_square(frameA)
        frameB = self.resize_to_square(frameB)

        stacked_pair = np.hstack((frameA, frameB))

        return stacked_pair


if __name__ == "__main__":

    with open(r'C:\Users\Mac Prible\repos\learn-opencv\all_bundles.pkl', 'rb') as f:
        all_bundles = pickle.load(f)

    # self.syncronizer.ports = [0,1,2]
    # pairs = [(i,j) for i,j in combinations(self.syncronizer.ports,2)]

    session = Session("test_session")
    session.load_cameras()
    session.load_rtds()
    # session.find_additional_cameras(
    syncr = Synchronizer(session, fps_target=6)

    stereo_cal = StereoCalibrator(syncr)

    # #%%
    # for frame_bundle in all_bundles:
    #     stereo_cal.process_frame_bundle(frame_bundle, time_threshold=.5, corner_threshold=6)
    #     stacked_pairs = stereo_cal.superframe(single_frame_height=250)

    #     cv2.imshow("Stereocalibration", stacked_pairs)
    #     time.sleep(.1)
    #     key = cv2.waitKey(1)
    #     if key == ord("q"):
    #         cv2.destroyAllWindows()
    #         break 

    
        
    cv2.destroyAllWindows()
    logging.debug(pprint.pformat(stereo_cal.stereo_inputs))
