
#%%
import pyxy3d.logger
import json

logger = pyxy3d.logger.get(__name__)
from pyxy3d import __root__
import pytest
import shutil
import cv2
from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyxy3d.trackers.tracker_enum import TrackerEnum
from pyxy3d.trackers.holistic_opensim_tracker import HolisticOpenSimTracker

def calculate_distance(xyz_trajectory_data:pd.DataFrame, point1:str, point2:str):
    """
    Given a set of xyz trajectories from tracked and triangulated landmarks, calculate the 
    mean distance between two named points, excluding outlier points for data cleanliness
    
    """
    # calculate the distance
    distances = np.sqrt((xyz_trajectory_data[point1 + '_x'] - xyz_trajectory_data[point2 + '_x']) ** 2 +
                     (xyz_trajectory_data[point1 + '_y'] - xyz_trajectory_data[point2 + '_y']) ** 2 +
                     (xyz_trajectory_data[point1 + '_z'] - xyz_trajectory_data[point2 + '_z']) ** 2)
    

        # calculate Q1, Q3 and IQR for outlier detection
    Q1 = distances.quantile(0.25)
    Q3 = distances.quantile(0.75)
    IQR = Q3 - Q1

    # filter out the outliers
    filtered_distances = distances[(distances >= Q1 - 1.5 * IQR) & (distances <= Q3 + 1.5 * IQR)]

    # calculate the average length
    average_length = filtered_distances.mean()
    return average_length


# symmetrical_measures = {
#     "Shoulder_Width":["left_shoulder", "right_shoulder"],
#     "Hip_Width":["left_hip", "right_hip"],
#     "Inner_Eye_Distance":["left_inner_eye", "right_inner_eye"]
# }

# bilateral_measures = {
#     "Hip_Shoulder_Distance":["hip", "shoulder"],
#     "Shoulder_Inner_Eye_Distance":["inner_eye", "shoulder"],
#     "Palm": ["index_finger_MCP", "pinky_MCP"],
#     "Foot":["heel", "foot_index"],  
#     "Upper_Arm":["shoulder","elbow"],
#     "Forearm":["elbow", "wrist"],
#     "Wrist_to_MCP1":["wrist", "thumb_MCP"],
#     "Wrist_to_MCP2":["wrist", "index_finger_MCP"],
#     "Wrist_to_MCP3":["wrist", "middle_finger_MCP"],
#     "Wrist_to_MCP4":["wrist", "ring_finger_MCP"],
#     "Wrist_to_MCP5":["wrist", "pinky_MCP"],
#     "Prox_Phalanx_1":["thumb_MCP", "thumb_IP"],
#     "Prox_Phalanx_2":["index_finger_MCP", "index_finger_PIP"],
#     "Prox_Phalanx_3":["middle_finger_MCP", "middle_finger_PIP"],
#     "Prox_Phalanx_4":["ring_finger_MCP", "ring_finger_PIP"],
#     "Prox_Phalanx_5":["pinky_MCP", "pinky_PIP"],
#     "Mid_Phalanx_2":["index_finger_PIP", "index_finger_DIP"],
#     "Mid_Phalanx_3":["middle_finger_PIP","middle_finger_DIP"],
#     "Mid_Phalanx_4":["ring_finger_PIP", "ring_finger_DIP"],
#     "Mid_Phalanx_5":["pinky_PIP", "pinky_DIP"],
#     "Dist_Phalanx_1":["thumb_IP", "thumb_tip"],
#     "Dist_Phalanx_2":["index_finger_DIP", "index_finger_tip"],
#     "Dist_Phalanx_3":["middle_finger_DIP","middle_finger_tip"],
#     "Dist_Phalanx_4":["ring_finger_DIP", "middle_finger_tip"],
#     "Dist_Phalanx_5":["pinky_DIP", "pinky_tip"],
#     "Thigh_Length":["hip","knee"],
#     "Shin_Length": ["knee", "ankle"]
# }
# usage 

if __name__ == "__main__":
    xyz_csv_path = Path(__root__,"tests", "reference", "auto_rig_config_data", "xyz_HOLISTIC_OPENSIM_labelled.csv")
    tracker = HolisticOpenSimTracker()

    xyz_trajectories = pd.read_csv(xyz_csv_path)
    json_path = Path(xyz_csv_path.parent, "autorig.json")

    # for testing purposes, need to make sure that this file is not there before proceeding
    json_path.unlink(missing_ok=True)
    assert not json_path.exists()
    
    autorig_config = {} # Dictionary that will become json

    # average distances across the bilateral measures
    for measure, points in bilateral_measures.items():
        logger.info(f"Calculating mean distance (IQR) for {measure}")

        mean_distance = 0
        for side in ["left", "right"]:
            
            point1 = f"{side}_{points[0]}"
            point2 = f"{side}_{points[1]}"

            distance = calculate_distance(xyz_trajectories, point1,point2)
            logger.info(f"Between {point1} and {point2} the mean distance is {distance}")     
            mean_distance += distance/2
        autorig_config[measure] = round(mean_distance,4)
           
           
    for measure, points in symmetrical_measures.items():
        distance = calculate_distance(xyz_trajectories, points[0],points[1])
        autorig_config[measure] = round(distance,4)
     
    with open(json_path,"w") as f:
        json.dump(autorig_config, f, indent=4)    
   
    # make sure you created the file 
    assert json_path.exists()

    with open(json_path,"r") as f:
        check_autorig_config = json.load(f)    
  
    # make sure all measures are accounted for and sensible  
    for measure, points in symmetrical_measures.items():
        assert measure in check_autorig_config.keys()
        assert type(check_autorig_config[measure]) == float

    for measure, points in bilateral_measures.items():
        assert measure in check_autorig_config.keys()
        assert type(check_autorig_config[measure]) == float