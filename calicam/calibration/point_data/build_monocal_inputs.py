#%%

import calicam.logger
logger = calicam.logger.get(__name__)

import cv2
import sys
import pandas as pd
from calicam import __root__
sys.path.insert(0,__root__)
from pathlib import Path
import numpy as np
import toml



# specify the session folder
session_path =  Path(__root__,"tests", "5_cameras")

config_path = Path(session_path,"config.toml")

#%%

ports = [0,1,2,3,4]

corner_count_threshold = 11
top_x_count = 9

port = 0
camera_resolution = [640,480]
point_data_path = Path(session_path, "recording",  "point_data.csv")

def get_monocal_data(port, camera_resolution, point_data_path, corner_count_threshold, top_x_count):
    """
    returns a dictionary o
    """
    #%%
    port = 0
    camera_resolution = [640,480]
    point_data_path = Path(session_path, "recording",  "point_data.csv")
    point_data = pd.read_csv(point_data_path)
    corner_count_threshold = 11
    top_x_count = 9

    points_by_multiport = (point_data
                        .filter(["sync_index", "point_id", "port"])
                        .pivot(index=["sync_index", "point_id"], columns="port", values="port")
                        .reset_index()
                        .fillna('')
    )

    def get_multiport(row, ports):
        """
        returns a string of the format "_0_1_2" for points which were captured by cameras 0,1 and 2.
        """
        text = ""
        for port in ports:
            label = row[port]
            if label != "":
                label = str(int(label))
                text = text + "_"+label
    
        return text

    points_by_multiport["captured_by"] = points_by_multiport.apply(get_multiport,axis=1, args=(ports,))

    single_port_points = (points_by_multiport
                        .loc[points_by_multiport[port]==port]
                        .assign(port=port)
    )

    board_counts = (single_port_points
                    .filter(["sync_index", "point_id"])
                    .groupby("sync_index")
                    .count()
                    .rename({"point_id":"corner_count"}, axis=1)
                    )
    
    board_seen_by = (single_port_points
                    .groupby(["port", "sync_index", "captured_by"])
                    .agg("count")
                    .rename({"point_id":"seen_by_count"}, axis=1)
                    .reset_index()
    )

    board_most_seen_by = (board_seen_by
                        .groupby(["port", "sync_index"])
                        .first()
                        .drop(columns="seen_by_count") # this no longer means much...only for one "seenby group"
                        .rename({"captured_by":"most_captured_by"}, axis=1)
                        .reset_index()
    )

    board_counts_most_seen_by = board_counts.merge(board_most_seen_by,"left", on=["sync_index"])

    criteria = board_counts_most_seen_by["corner_count"] >= corner_count_threshold
    board_counts_most_seen_by = board_counts_most_seen_by[criteria]
    board_counts_most_seen_by = (board_counts_most_seen_by
                                #  .reset_index()
                                .groupby("most_captured_by")
                                .head(top_x_count)
                                .reset_index()
                                .sort_values(["most_captured_by"]))

    port_monocal_data = point_data.merge(board_counts_most_seen_by,"right", ["sync_index", "port"])
    
#%%
    
    return port_monocal_data    

  


def calibrate(port, resolution, port_monocal_data):

    """
    Use the recorded image corner positions along with the objective
    corner positions based on the board definition to calculated
    the camera matrix and distortion parameters
    """
    
    #%%
    resolution = (640,480)
    sync_indices = port_monocal_data["sync_index"].to_numpy().round().astype(int)
    img_loc_x = port_monocal_data["img_loc_x"].to_numpy().astype(np.float32)
    img_loc_y = port_monocal_data["img_loc_y"].to_numpy().astype(np.float32)
    board_loc_x = port_monocal_data["board_loc_x"].to_numpy().astype(np.float32)
    board_loc_y = port_monocal_data["board_loc_y"].to_numpy().astype(np.float32)
    board_loc_z = board_loc_x*0 # all on the same plane
    
    # build the actual inputs for the calibration...
    img_x_y = np.vstack([img_loc_x, img_loc_y]).T
    board_x_y_z = np.vstack([board_loc_x, board_loc_y, board_loc_z]).T

    import time
    print(time.time())
    img_locs = [] #  np.ndarray([])
    board_locs = [] #np.ndarray([])
    for sync_index in np.unique(sync_indices):
        same_frame = sync_indices==sync_index
        # np.hstack([img_locs, img_x_y[same_frame]])
        # np.hstack([board_locs, board_x_y_z[same_frame]])
        img_locs.append(img_x_y[same_frame])
        board_locs.append(board_x_y_z[same_frame])
    
    print(time.time())

    logger.info(f"Calibrating camera {port}....")
    error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        board_locs, img_locs, resolution, None, None
    )
    logger.info(f"Error: {error}")
    logger.info(f"Camera Matrix: {mtx}")
    logger.info(f"Distortion: {dist}")


from calicam.session import Session
session = Session(session_path)





# board_locs = np.array(board_locs, dtype='object')

# %%
print(f"Using {len(img_locs)} board captures to calibrate camera....")
start = time.time()
calibrate(port = port, resolution=resolution, img_loc=img_locs, board_loc=board_locs )
elapsed = time.time()-start
print(f"{elapsed} seconds elapsed to perform one camera calibration")

# %%
