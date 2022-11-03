import pickle
import cv2
import logging
import time
from itertools import combinations
import numpy as np
import imutils
import pprint


logging.basicConfig(filename="stereocalibration.log", 
                    filemode = "w", 
                    # level=logging.INFO)
                    level=logging.DEBUG)



class StereoCalibrator:

    def __init__(self, ports):

        self.ports = ports
        self.pairs = [(i,j) for i,j in combinations(ports,2)]

        # self.frame_bundle = frame_bundle
        # initialize dictionary to hold stereo inputs
        blank_dict = {"obj":[], "img_A": [], "img_B": [], "frame_time": []}
        self.stereo_inputs = {pair:blank_dict for pair in pairs}
        self.last_store_time = time.perf_counter()        

    def process_frame_bundle(self, frame_bundle, time_threshold=0.5, corner_threshold=7):
        self.frame_bundle = frame_bundle
        # self.corner_threshold = corner_threshold

        for pair in self.pairs:
            portA = pair[0]
            portB = pair[1]
            self.set_common_ids(pair[0], pair[1])
            enough_corners = len(self.common_ids) > corner_threshold
            enough_time = time.perf_counter() - self.last_store_time > time_threshold

            if enough_corners and enough_time:
                obj, img_A = self.get_obj_img_points(portA)
                _, img_B = self.get_obj_img_points(portB)

             #    if len(self.common_ids) > self.corner_threshold:
                self.stereo_inputs[pair]["obj"].append(obj)
                self.stereo_inputs[pair]["img_A"].append(img_A)
                self.stereo_inputs[pair]["img_B"].append(img_B)
                self.last_store_time = time.perf_counter()

    def superframe(self, single_frame_height=250):

        self.single_frame_height = single_frame_height
        stacked_bundle = np.array([])

        for pair in self.pairs:

            portA = pair[0]
            portB = pair[1]        

            # create stacked frames to simplify video output
            hstacked_pair = self.hstack_frames(portA, portB)

            if stacked_bundle.any():
                stacked_bundle = np.vstack((stacked_bundle, hstacked_pair))
            else:
                stacked_bundle = hstacked_pair

        return stacked_bundle

    def resize_to_square(self, port):
        """ returns a square image with black borders to round it out. If the 
        data is none, then makes the image completely blank"""

        edge = self.single_frame_height
        bundle = self.frame_bundle[port]
        if bundle is None:
            frame = np.zeros((edge, edge, 3), dtype=np.uint8)

        else:
            frame = self.frame_bundle[port]["frame"]

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

        # if self.frame_bundle[port] and len(self.common_ids)>self.corner_threshold:
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


    def hstack_frames(self, portA, portB):

        frameA = self.resize_to_square(portA)
        frameB = self.resize_to_square(portB)

        stacked_pair = np.hstack((frameA, frameB))

        return stacked_pair


if __name__ == "__main__":

    with open(r'C:\Users\Mac Prible\repos\learn-opencv\all_bundles.pkl', 'rb') as f:
        all_bundles = pickle.load(f)

    ports = [0,1,2]
    pairs = [(i,j) for i,j in combinations(ports,2)]

    
    stereo_cal = StereoCalibrator(ports)
    #%%
    for frame_bundle in all_bundles:
        stereo_cal.process_frame_bundle(frame_bundle, time_threshold=1, corner_threshold=7)
        stacked_pairs = stereo_cal.superframe(single_frame_height=250)

        cv2.imshow("Stereocalibration", stacked_pairs)
        cv2.waitKey(1)
        time.sleep(.03 )

    cv2.destroyAllWindows()
    logging.debug(pprint.pformat(stereo_cal.stereo_inputs))
