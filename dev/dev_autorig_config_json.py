
#%%
import pyxy3d.logger

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



def calculate_distance(xyz_trajectory_data:pd.DataFrame, point1:str, point2:str):
    """
    
    
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
    # print(f"Average Length: {average_length}")
    # histogram of the distances
    # plt.hist(filtered_distances, bins=30, alpha=0.8)
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Distances')
    # plt.show()
    return average_length


symmetrical_measures = {
    "Shoulder_Width":["left_shoulder", "right_shoulder"],
    "Hip_Width":["left_hip", "right_hip"],
    "Inner_Eye_Distance":["left_inner_eye", "right_inner_eye"]
}

averaged_measures = {
    "Hip_Shoulder_Distance":["hip", "shoulder"],
    "Shoulder_Inner_Eye_Distance":["inner_eye", "shoulder"]
}

bilateral_measures = {
    "Palm": ["index_finger_MCP", "pinky_MCP"],
    "Foot":["heel", "foot_index"],  
    "Upper_Arm":["shoulder","elbow"],
    "Forearm":["elbow", "wrist"],
    "Wrist_to_MCP1":["wrist", "thumb_MCP"],
    "Wrist_to_MCP2":["wrist", "index_finger_MCP"],
    "Wrist_to_MCP3":["wrist", "middle_finger_MCP"],
    "Wrist_to_MCP4":["wrist", "ring_finger_MCP"],
    "Wrist_to_MCP5":["wrist", "pinky_MCP"],
    "Prox_Phalanx_1":["thumb_MCP", "thumb_IP"],
    "Prox_Phalanx_2":["index_finger_MCP", "index_finger_PIP"],
    "Prox_Phalanx_3":["middle_finger_MCP", "middle_finger_PIP"],
    "Prox_Phalanx_4":["ring_finger_MCP", "ring_finger_PIP"],
    "Prox_Phalanx_5":["pinky_MCP", "pinky_PIP"],
    "Mid_Phalanx_2":["index_finger_PIP", "index_finger_DIP"],
    "Mid_Phalanx_3":["middle_finger_PIP","middle_finger_DIP"],
    "Mid_Phalanx_4":["ring_finger_PIP", "ring_finger_DIP"],
    "Mid_Phalanx_5":["pinky_PIP", "pinky_DIP"],
    "Dist_Phalanx_1":["thumb_IP", "thumb_tip"],
    "Dist_Phalanx_2":["index_finger_DIP", "index_finger_tip"],
    "Dist_Phalanx_3":["middle_finger_DIP","middle_finger_tip"],
    "Dist_Phalanx_4":["ring_finger_DIP", "middle_finger_tip"],
    "Dist_Phalanx_5":["pinky_DIP", "pinky_tip"],
    "Thigh_Length":["hip","knee"],
    "Shin_Length": ["knee", "ankle"]
}
# usage 

if __name__ == "__main__":
    xyz_csv_path = Path(__root__,"tests", "reference", "auto_rig_config_data", "xyz_HOLISTIC_OPENSIM_labelled.csv")

    xyz_trajectories = pd.read_csv(xyz_csv_path)

    for measure, points in bilateral_measures.items():
        logger.info(f"Calculating mean distance (IQR) for {measure}")

        for side in ["left", "right"]:
            
            point1 = f"{side}_{points[0]}"
            point2 = f"{side}_{points[1]}"

            distance = calculate_distance(xyz_trajectories, point1,point2)
            logger.info(f"Between {point1} and {point2} the mean distance is {distance}")     
        
            
