

#%%
# This is a scratchpad for me to work through the data processing to transform points.csv 
# into a PointHistory data object and then perform stereotriangulation on it given a CameraArray
from itertools import combinations
import pandas as pd
import numpy as np
from pathlib import Path
from calicam import __root__
from calicam.cameras.data_packets import PointPacket, FramePacket, SyncPacket
#%%

point_data_path = Path(__root__, "tests", "5_cameras", "recording", "point_data.csv")
point_data = pd.read_csv(point_data_path)

xy_sync_indices = point_data["sync_index"].to_numpy()
xy_camera_indices = point_data["port"].to_numpy()
xy_point_id = point_data["point_id"].to_numpy()

img_x = point_data["img_loc_x"].to_numpy()
img_y = point_data["img_loc_y"].to_numpy()
xy_img = np.vstack([img_x,img_y]).T


ports = np.unique(xy_camera_indices)
sync_indices = np.unique(xy_sync_indices)

pairs = [(i,j) for i,j in combinations(ports,2) if i<j]

for sync_index in [0,1,2,3]: # sync_indices:
    port_points = point_data.query(f"sync_index == {sync_index}")

    ports = port_points["port"].unique()
    print(f"{sync_index}: {ports}")

    frame_packets = {}

    for port in ports:
        points = port_points.query(f"port == {port}")
        frame_time = points["frame_time"].iloc[0]
        frame_index = points["frame_index"].iloc[0]

        point_id = points["point_id"]

        img_loc_x = points["img_loc_x"].to_numpy()        
        img_loc_y = points["img_loc_y"].to_numpy()        
        img_loc = np.vstack([img_loc_x,img_loc_y]).T

        board_loc_x = points["board_loc_x"].to_numpy()        
        board_loc_y = points["board_loc_y"].to_numpy()        
        board_loc = np.vstack([board_loc_x,board_loc_y]).T

        point_packet = PointPacket(point_id, img_loc, board_loc)
        frame_packet = FramePacket(port, frame_time, None, frame_index, point_packet)
        frame_packets[port] = frame_packet
        
    sync_packet = SyncPacket(sync_index, frame_packets)        

# %%
