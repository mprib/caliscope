#%%
from pathlib import Path

import sys
import cv2
import numpy as np
import pandas as pd

repo = str(Path(__file__)).split("src")[0]
sys.path.insert(0, repo)

from src.cameras.camera_array import CameraArray, CameraArrayBuilder

# Going to start pulling in variables from the csv files that are saved out and
# attempt to get the data formatted in the same way that is required by the
# sample code from scipy.org

session_directory = Path(repo, "sessions", "iterative_adjustment")
config_file = Path(session_directory, "config.toml")

array_builder = CameraArrayBuilder(config_file)

camera_array = array_builder.get_camera_array()

#%%

n_cameras = len(camera_array.cameras)
print(f"Number of cameras: {n_cameras}")


# for each camera build the 9 element parameter index
# camera_params with shape (n_cameras, 9) contains initial estimates of parameters for all cameras.
# First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula),
# next 3 components form a translation vector, then a focal distance and two distortion parameters.
# note that the distortion parameters only reflect the radial distortion (not the tangential)
camera_params = None
for port, cam in camera_array.cameras.items():

    # pull rotation matrix and convert to Rodrigues
    rotation_matrix = camera_array.cameras[port].rotation
    rotation_rodrigues = cv2.Rodrigues(rotation_matrix)[0]  # elements 0,1,2
    translation = camera_array.cameras[port].translation  # elements 3,4,5

    # two focal lengths for potentially rectangular pixels...
    # I'm assuming they are square
    fx = camera_array.cameras[port].camera_matrix[0, 0]
    fy = camera_array.cameras[port].camera_matrix[1, 1]
    f = (fx + fy) / 2  # element 6

    # get k1 and k2 from distortion
    # note that a future plan my pre-undistort the image
    # and hard code this to zero within the `project` function
    # this will reduce parameter estimates and still account for
    # intrinsics (I think)
    k1 = camera_array.cameras[port].distortion[0, 0]  # element 7
    k2 = camera_array.cameras[port].distortion[0, 1]  # element 8

    port_param = np.hstack([rotation_rodrigues[:, 0], translation[:, 0], f, k1, k2])

    if camera_params is None:
        camera_params = port_param
    else:
        camera_params = np.vstack([camera_params, port_param])

    print("--------")
    print(camera_params)

# Get 3d points that will be used for bundle adjustment
# this will only include those points that were observed simultaneously by
# more than one pair of cameras. In the three camera case, it will mean
# observed by all three cameras, which I think will be a fairly small number

#%%
points_3d_csv_path = Path(session_directory, "triangulated_points.csv")
points_3d = pd.read_csv(points_3d_csv_path)


points_3d = (
    points_3d[["bundle", "id", "pair", "x_pos", "y_pos", "z_pos"]]
    .groupby(["bundle", "id"])
    .agg({"x_pos": "mean", "y_pos": "mean", "z_pos": "mean", "id": "size"})
    .rename(columns={"id": "count"})
    .query("count > 1")
)

points_3d
# %%
# Get points by camera...go back to paired point data

# camera_id
# camera_id with shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.


# point_ind
# point_ind with shape (n_observations,) contains indices of points (from 0 to n_points - 1) involved in each observation.


# points_2d
# points_2d with shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each observations.


# %%
# points_3d
# points_3d with shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
# %%
