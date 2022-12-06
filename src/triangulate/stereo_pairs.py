# create a json to hold all possible ways of storing pairs
# this will be used to build a comprehensive set of visualizations that can
# aid with future camera system calibrations
#%%
import pandas as pd
import toml
import numpy as np
import math
import sys
from pathlib import Path

# for running in interactive jupyter
sys.path.insert(0,Path(__file__).parent.parent)

# from a set of stereocalibrations, build global configuration of camera locations
# begin by making a data table that will allow easier sorting and configuration

def rotationMatrixToEulerAngles(R) :
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def rotation_to_float(rotation_matrix):
    """Convert the text rotation matrix stored in config.toml to an array 
    that can be converted into euler angles"""
    new_matrix = []
    for row in rotation_matrix:
        new_row = [float(x) for x in row]
        new_matrix.append(new_row)

    return np.array(new_matrix, dtype=np.float32)


config_path = Path(Path(__file__).parent.parent, r"config 2.toml")
config = toml.load(str(config_path))

stereo_params = None

# get all pairs to iterate over
pairs = []
for key, params in config.items():
    if "stereo" in key:
        pair = key.split("_")[1:3]
        if pair not in pairs:
            pairs.append(pair) 

        
for pair in pairs:
    new_row = pd.DataFrame({"PrimaryCam":pair[0], "SecondaryCam": pair[1]}, index=[0])
    text_key = f"stereo_{pair[0]}_{pair[1]}" 
    new_row["RMSE"] = config[f"{text_key}_RMSE"]


    translation = config[f"{text_key}_translation"]

    # currently these are list of lists...might try to clean that up 
    # at some point, but otherwise here is this thing

    try:
        translation = [x[0] for x in translation]
    except:
        pass
    new_row["trans_x"] = float(translation[0])
    new_row["trans_y"] = float(translation[1])
    new_row["trans_z"] = float(translation[2])

    rotation = config[f"{text_key}_rotation"]
    rotation = rotation_to_float(rotation)
    rotation = rotationMatrixToEulerAngles(rotation)
    rotation = [x*(180/math.pi) for x in rotation] # convert to degrees
    new_row["rot_x"] = rotation[0]
    new_row["rot_y"] = rotation[1]
    new_row["rot_z"] = rotation[2]

    if stereo_params is None:
        stereo_params = new_row
    else:
        stereo_params = pd.merge(stereo_params,new_row, how="outer")

#%%


# create an extra set of parameters that has an inverted frame of reference
rows = stereo_params.shape[0]
for index in range(rows):
    orig_row = stereo_params.loc[index]

    new_row = pd.DataFrame({}, index=[0])
    new_row["PrimaryCam"] = orig_row["SecondaryCam"]
    new_row["SecondaryCam"] = orig_row["PrimaryCam"]

    new_row["RMSE"] = orig_row["RMSE"]
    new_row["trans_x"] = -orig_row["trans_x"]
    new_row["trans_y"] = -orig_row["trans_y"]
    new_row["trans_z"] = -orig_row["trans_z"]
    new_row["rot_x"] = -orig_row["rot_x"]
    new_row["rot_y"] = -orig_row["rot_y"]
    new_row["rot_z"] = -orig_row["rot_z"]

    stereo_params = pd.merge(stereo_params,new_row, how="outer")


# %%
print(stereo_params)

# next up is sorting through the cameras to get the primary with the lowest 
# RMSE error

stereo_params = stereo_params.sort_values(["PrimaryCam", "SecondaryCam"])

# (
#     stereo_params
#     .sort_value(["PrimaryCam", "SecondaryCam"])
    

# )