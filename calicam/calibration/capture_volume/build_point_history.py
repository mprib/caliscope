#%%

# This is a scratchpad for me to work through the data processing to transform points.csv 
# into a PointHistory data object and then perform stereotriangulation on it given a CameraArray
from itertools import combinations
import pandas as pd
import numpy as np
from pathlib import Path
from calicam import __root__
#%%

point_data_path = Path(__root__, "tests", "5_cameras", "recording", "point_data.csv")
point_data = pd.read_csv(point_data_path)


#%%
#### Get basic components of the PointHistory
xy_sync_indices = point_data["sync_index"].to_numpy()
xy_camera_indices = point_data["port"].to_numpy()
xy_point_id = point_data["point_id"].to_numpy()

img_x = point_data["img_loc_x"].to_numpy()
img_y = point_data["img_loc_y"].to_numpy()
xy_img = np.vstack([img_x,img_y]).T


ports = np.unique(xy_camera_indices)
sync_indices = np.unique(xy_sync_indices)

pairs = [(i,j) for i,j in combinations(ports,2) if i<j]
#%%


# %%

def get_paired_points(pair):
    sync_points_by_port = (point_data
                    .filter(["sync_index", "point_id", "port"])
                    .pivot(index=['sync_index', 'point_id'], columns='port', values='port').notna()
                    .reset_index())
    port_A = pair[0]
    port_B = pair[1]

    # identify the points that are scene by the pair of cameras
    common_sync_points = sync_points_by_port.loc[sync_points_by_port[port_A] & sync_points_by_port[port_B]]

    paired_points = ( point_data
        .query(f"port == {pair[0]} or port == {pair[1]}")
        .merge(common_sync_points,"right", ["sync_index", "point_id"])
        .filter(["sync_index", "port", "point_id", "img_loc_x", "img_loc_y"])
        .pivot(index=["sync_index", "point_id"], columns="port")
        .reset_index()
    )
    
    paired_points.columns = paired_points.columns.map(lambda x: f"{x[0]}_{str(x[1])}")
    
    img_loc_x_A = paired_points[f"img_loc_x_{port_A}"]   
    img_loc_y_A = paired_points[f"img_loc_y_{port_A}"]   

    img_loc_x_B = paired_points[f"img_loc_x_{port_B}"]   
    img_loc_y_B = paired_points[f"img_loc_y_{port_B}"]   

    
    return paired_points
#%%
import time
print(time.time())
all_paired_points = {}
for pair in pairs:
    print(pair)
    all_paired_points[pair] = get_paired_points(pair)
print(time.time())
# %%
pair = (0,2)
paired_points = get_paired_points(pair)


# %%
point_data_port_pivot = (point_data
                        .filter(["sync_index", "port", "point_id", "img_loc_x", "img_loc_y"])
                        .pivot(index=["sync_index", "point_id"], columns="port")
                        .reset_index()
    )

point_data_port_pivot.columns = point_data_port_pivot.columns.map(lambda x: f"{x[0]}_{str(x[1])}")

# all of the 2D points by 3D point
point_data_port_pivot = (point_data_port_pivot.fillna(""))
# %%