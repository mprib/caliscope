#%%


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
# from src.cameras.camera_array import ArrayDiagnosticData
from src.calibration.bundle_adjustment.point_data import PointData
from src.calibration.bundle_adjustment.calibration_diagnostics import (
    get_charuco,
    create_summary_df,
    get_diagnostic_data,
    get_corners_xyz,
)


# calibration_directory = Path(repo, "sessions", "iterative_adjustment", "recording")
calibration_directory = Path(repo, "sessions", "default_res_session", "recording")
before_path = Path(calibration_directory, "before_bund_adj.pkl")
after_path = Path(calibration_directory, "after_bund_adj.pkl")

before = get_diagnostic_data(before_path)
after = get_diagnostic_data(after_path)

before_df = create_summary_df(before_path, "before")
after_df = create_summary_df(after_path, "after")

before_and_after = pd.concat([before_df, after_df])

# Get array of chessboard_ids and locations in a board frame of referencefrom before_and_after
config_path = Path(calibration_directory.parent, "config.toml")
corners_3d = get_corners_xyz(config_path, before_path,"before")


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
sync_indices = corners_3d["sync_index"].to_numpy()
unique_sync_indices = np.unique(sync_indices)

obj_id = corners_3d["obj_id"].to_numpy()

for x in unique_sync_indices:
    frame_corners = obj_id[sync_indices==x]
    all_pairs = cartesian_product(frame_corners,frame_corners)
    print(all_pairs)

charuco = get_charuco(config_path)
corner_count = charuco.board.chessboardCorners.shape[0]
board_ids = np.arange(corner_count)
# %%
