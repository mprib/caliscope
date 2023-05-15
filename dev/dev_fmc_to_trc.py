# The purpose of this code module is to take the numpy output of freemocap and convert
# it to a human readable format that will become the hub of translation to other formats
# additionally, it will generate a .trc file that can be used for opensim scaling
# and inverse kinematics 

import os
import csv
from pathlib import Path 
import numpy as np
import pandas as pd
import json
import xml.dom.minidom as md


mean_frame_rate = 1/30

# tracked landmarks will be used to label trajectory array and 
# convert to human readable csv
tracked_landmarks = get_landmark_index(pose_tracked)
tracked_landmark_count = len(tracked_landmarks)
trajectories = get_trajectory_dataframe()
            
# modelled trajectories will be used to create the trc and ignore
# tracked landmarks that don't get included in the osim model
model_landmarks = get_model_landmarks()



all_trajectories = get_trajectory_array()



# add in Frame Number and Time stamp 
merged_trajectories["Frame"] = [str(i) for i in range(0, len(merged_trajectories))]
merged_trajectories["Time"] = merged_trajectories["Frame"].astype(float) / float(mean_frame_rate)        

# get the correct order for all dataframe columns
column_order = []
for marker in tracked_landmarks:
    column_order.append(marker + "_x")
    column_order.append(marker + "_y")
    column_order.append(marker + "_z")

# Add Frame and Time
column_order.insert(0, "Frame")
column_order.insert(1, "Time")

# reorder the dataframe, note frame number in 0 position remains
merged_trajectories = merged_trajectories.reindex(columns=column_order)

        
num_markers = len(model_landmarks)
data_rate= mean_frame_rate # not sure how this is different from camera rate
mean_frame_rate= mean_frame_rate
units = 'm'
orig_data_rate = 25
orig_data_start_frame = 0

trajectories_for_trc = trajectories

# make a list of the trajectories to keep
# only those that are being modelled in osim
# 
keep_trajectories = []
for lm in model_landmarks:
    keep_trajectories.append(lm+"_x")
    keep_trajectories.append(lm+"_y")
    keep_trajectories.append(lm+"_z")

# if an osim has markers, only keep those markers,
# otherwise export everything for inspection
if keep_trajectories:
    trajectories_for_trc = trajectories_for_trc[keep_trajectories]
        
if drop_na:
    trajectories_for_trc = trajectories_for_trc.dropna()
        

# these are at top of .trc
orig_num_frames = len(trajectories_for_trc) - 1
num_frames = orig_num_frames

trc_path = os.path.join(trc_filename)

# this will create the formatted .trc file
with open(trc_path, 'wt', newline='', encoding='utf-8') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(["PathFileType",
                        "4", 
                        "(X/Y/Z)",	
                        trc_filename])
    tsv_writer.writerow(["DataRate",
                        "CameraRate",
                        "NumFrames",
                        "NumMarkers", 
                        "Units",
                        "OrigDataRate",
                        "OrigDataStartFrame",
                        "OrigNumFrames"])
    tsv_writer.writerow([data_rate, 
                        mean_frame_rate,
                        num_frames, 
                        num_markers, 
                        units, 
                        orig_data_rate, 
                        orig_data_start_frame, 
                        orig_num_frames])

    # create names of trajectories, skipping two columns (top of table)
    header_names = ['Frame#', 'Time']
    for trajectory in model_landmarks:
        header_names.append(trajectory)
        header_names.append("")
        header_names.append("")    

    tsv_writer.writerow(header_names)

    # create labels for x,y,z axes (below landmark names in header)
    header_names = ["",""]
    for i in range(1,len(model_landmarks)+1):
        header_names.append("X"+str(i))
        header_names.append("Y"+str(i))
        header_names.append("Z"+str(i))

    tsv_writer.writerow(header_names)

    # the .trc fileformat expects a blank fourth line
    tsv_writer.writerow("")

    # add in frame and Time stamp to the data frame 
    # this redundent step is due to potentially dropping frames earlier
    # when pairing down the dataframe to only relevant markers
    if keep_trajectories:
        trajectories_for_trc.insert(0, "Frame", [str(i) for i in range(0, len(trajectories_for_trc))])
        trajectories_for_trc.insert(1, "Time", trajectories_for_trc["Frame"].astype(float) / float(mean_frame_rate))
            
    # and finally actually write the trajectories
    for row in range(0, len(trajectories_for_trc)):
        tsv_writer.writerow(trajectories_for_trc.iloc[row].tolist())


