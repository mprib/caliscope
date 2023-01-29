#%%


import seaborn as sns
from time import perf_counter
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd

# some convenient reference paths
repo = str(Path.cwd()).split("src")[0]
# update path
sys.path.insert(0, repo)
# which enables import of relevant class
# from calicam.cameras.camera_array import ArrayDiagnosticData
from calicam.calibration.bundle_adjustment.point_data import PointData
from calicam.calibration.bundle_adjustment.calibration_diagnostics import (
    get_charuco,
    create_summary_df,
    get_diagnostic_data,
    get_corners_xyz,
)


# calibration_directory = Path(repo, "sessions", "iterative_adjustment", "recording")
calibration_directory = Path(repo, "sessions", "default_res_session", "recording")
before_path = Path(calibration_directory, "before_bund_adj.pkl")
after_path = Path(calibration_directory, "after_bund_adj.pkl")

# before = get_diagnostic_data(before_path)
# after = get_diagnostic_data(after_path)

before_df = create_summary_df(before_path, "before")
after_df = create_summary_df(after_path, "after")


# Get array of chessboard_ids and locations in a board frame of referencefrom before_and_after
config_path = Path(calibration_directory.parent, "config.toml") 
corners_3d = get_corners_xyz(config_path,after_path,"after")

#%%
sns.relplot(data = after_df, 
            x = "reproj_error_x",
            y = "reproj_error_y",
            col="camera",
            # row= "label"
            kind="scatter")

#%%
sns.displot(
    data=after_df,
    x="reproj_error",
    col="camera"
)

#%%

def cartesian_product(*arrays):
    """
    https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)
    

# %%

def get_paired_obj_indices(corners_3d: pd.DataFrame):
    """given a dataframe that contains all observed charuco corners across sync_indices, 
    return a Nx2 matrix of paired object indices that will represent all possible
    joined lines between charuco corners"""

    # get columns out from data frame for numpy calculations
    sync_indices = corners_3d["sync_index"].to_numpy(dtype=np.int32)
    unique_sync_indices = np.unique(sync_indices)
    obj_id = corners_3d["obj_id"].to_numpy(dtype=np.int32)

    start = perf_counter()

    # for a given sync index (i.e. one board snapshot) get all pairs of object ids
    paired_obj_indices = None
    for x in unique_sync_indices:
        sync_obj = obj_id[sync_indices==x] #objects (corners) that all occur on the same frame
        all_pairs = cartesian_product(sync_obj,sync_obj)
        if paired_obj_indices is None:
            paired_obj_indices = all_pairs
        else:
            paired_obj_indices = np.vstack([paired_obj_indices,all_pairs])
        # print(all_pairs)
    
    # paired_corner_indices will contain duplicates (i.e. [0,1] and [1,0]) as well as self-pairs ([0,0], [1,1])
    # this need to get filtered out
    reformatted_paired_obj_indices = np.zeros(paired_obj_indices.shape,dtype=np.int32)
    reformatted_paired_obj_indices[:,0] = np.min(paired_obj_indices,axis=1) # smaller on left
    reformatted_paired_obj_indices[:,1] = np.max(paired_obj_indices,axis=1) # larger on right
    reformatted_paired_obj_indices = np.unique(reformatted_paired_obj_indices,axis=0)
    reformatted_paired_obj_indices = reformatted_paired_obj_indices[reformatted_paired_obj_indices[:,0]!=reformatted_paired_obj_indices[:,1]]

    stop = perf_counter()
    elapsed = stop - start
    print(f"Time to create paired object indices: {round(elapsed,5)} sec")
    return reformatted_paired_obj_indices
#%%

paired_obj_indices = get_paired_obj_indices(corners_3d)
#%%

charuco = get_charuco(config_path)
corner_count = charuco.board.chessboardCorners.shape[0]
board_ids = np.arange(corner_count)
corner_ids = corners_3d["charuco_id"]
corners_board_xyz = charuco.board.chessboardCorners[corner_ids]

#%%

corners_world_xyz = corners_3d[["obj_x", "obj_y", "obj_z"]].to_numpy()

corners_world_A = corners_world_xyz[paired_obj_indices[:,0]]
corners_world_B = corners_world_xyz[paired_obj_indices[:,1]]
corners_board_A = corners_board_xyz[paired_obj_indices[:,0]]
corners_board_B = corners_board_xyz[paired_obj_indices[:,1]]


#%%
start = perf_counter()
distance_world_A_B = np.sqrt(np.sum((corners_world_A-corners_world_B) ** 2,axis=1))
distance_board_A_B = np.sqrt(np.sum((corners_board_A-corners_board_B) ** 2,axis=1))

distance_world_A_B = np.round(distance_world_A_B,5)
distance_board_A_B = np.round(distance_board_A_B,5)

distance_error = distance_world_A_B-distance_board_A_B

stop = perf_counter()
print(f"Time to calculate distances across all possible paired corners is {stop-start}")

# %%
start = perf_counter()
distance_error_df = pd.DataFrame(distance_error,columns=["Distance_Error"])
stop = perf_counter()
print(f"Time to convert array to dataframe is {stop-start}")
#
# %%
start = perf_counter()
distance_error_df["Distance_Error_mm"] = distance_error_df["Distance_Error"]*1000
stop = perf_counter()
print(f"Time to multiply by 1,000:  {stop-start}")

# %%

sns.displot(data=distance_error_df,x="Distance_Error_mm")
# %%
distance_error_df["Distance_Error_mm_abs"] = abs(distance_error_df["Distance_Error_mm"])
distance_error_df.describe()

# %%
