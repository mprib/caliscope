import pickle
import cv2
import logging
import time
from itertools import combinations
import numpy as np
import imutils
logging.basicConfig(filename="stereocalibration.log", 
                    filemode = "w", 
                    # level=logging.INFO)
                    level=logging.DEBUG)

with open(r'C:\Users\Mac Prible\repos\learn-opencv\all_bundles.pkl', 'rb') as f:
    all_bundles = pickle.load(f)
# %%
ports = [0,1,2]
pairs = [(i,j) for i,j in combinations(ports,2)]
# %%

def square_resize(frame, final_side_length):
    frame = cv2.flip(frame,1)

    # pad with 0s to make square and black
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

    frame = imutils.resize(frame, height=final_side_length)

    return frame    

def common_ids(portA, portB):
    corner_idsA = frame_bundle[portA]["corner_ids"]
    corner_idsB = frame_bundle[portB]["corner_ids"]
    common_corners = np.intersect1d(corner_idsA,corner_idsB)
    common_corners = common_corners.tolist()

    return common_corners

def obj_img_points(port, common_corner_ids):

    corner_ids = frame_bundle[port]["corner_ids"]
    frame_corners = frame_bundle[port]["frame_corners"].squeeze().tolist()
    board_FOR_corners = frame_bundle[port]["board_FOR_corners"].squeeze().tolist()   

    obj_points = []
    img_points = []

    for crnr_id, img, obj in zip(corner_ids, frame_corners, board_FOR_corners):
        if crnr_id in common_corner_ids:
            img_points.append(img)
            obj_points.append(obj)

    return obj_points, img_points 

print(pairs)
frame_height = 250
#%%
for frame_bundle in all_bundles:

    stacked_bundle = np.array([])

    for pair in pairs:
        portA = pair[0]
        portB = pair[1]        
        
        # create stacked frames to simplify video output
        frameA = frame_bundle[portA]["frame"]
        frameB = frame_bundle[portB]["frame"]

        frameA = square_resize(frameA, frame_height)
        frameB = square_resize(frameB, frame_height)

        stacked_pair = np.hstack((frameA, frameB))

        if stacked_bundle.any():
            stacked_bundle = np.vstack((stacked_bundle, stacked_pair))
        else:
            stacked_bundle = stacked_pair

        
        # get common corners between portA and portB
        in_common = common_ids(portA, portB)
        obj_A, img_A = obj_img_points(portA, in_common)
        obj_B, img_B = obj_img_points(portB, in_common)


        if len(in_common) > 6 and len(in_common)< 8:
            logging.debug(f"Common corners for pair {pair} are {in_common}")
            logging.debug(f"Objective location of points is {obj_A}")
            logging.debug(f"Should be same as  {obj_B}")
            logging.debug(f"Image Points for port A are {img_A}")
            logging.debug(f"Image Points for port B are {img_B}")

    cv2.imshow("Stereocalibration", stacked_bundle)

    cv2.waitKey(1)

    # time.sleep(.001 )

    # for port, frame_data in frame_bundle.items():
    #     print(port)

    #     if frame_data:
    #         print(f'showing frame data for port {port}')

    #         cv2.imshow(f"Port {port}", frame_data["frame"]) 

    #         cv2.waitKey(1)
