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

# points_3d with shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
#%%
points_3d_csv_path = Path(session_directory, "triangulated_points.csv")
points_3d_df = pd.read_csv(points_3d_csv_path)

# select only the 3d points that are shared across more than one pair of cameras
points_3d_df = (
    points_3d_df[["bundle", "id", "pair", "x_pos", "y_pos", "z_pos"]]
    .sort_values(["bundle", "id"])
    .groupby(["bundle", "id"])
    .agg({"x_pos": "mean", "y_pos": "mean", "z_pos": "mean", "pair": "size"})
    .rename(
        columns={"pair": "count", "x_pos": "x_3d", "y_pos": "y_3d", "z_pos": "z_3d"}
    )
    .query("count > 1")
    .drop(["count"], axis=1)
    .reset_index()
    .reset_index()
    .rename(columns={"index":"index_3d"})
)

# %%
# Convert paired_points_csv into a format that will be amenable to the bundle adjustment.
# This may end up being rather convoluted but I think is for more time efficient (for me)
# than going back and figuring out how to create these files live during processing
paired_point_csv_path = Path(session_directory, "paired_point_data.csv")
paired_points = pd.read_csv(paired_point_csv_path)

paired_points_A = paired_points[
    ["port_A", "bundle_index", "point_id", "loc_img_x_A", "loc_img_y_A"]
].rename(
    columns={
        "bundle_index": "bundle",
        "port_A": "camera",
        "point_id": "id",
        "loc_img_x_A": "x_2d",
        "loc_img_y_A": "y_2d",
    }
)

paired_points_B = paired_points[
    ["port_B", "bundle_index", "point_id", "loc_img_x_B", "loc_img_y_B"]
].rename(
    columns={
        "bundle_index": "bundle",
        "port_B": "camera",
        "point_id": "id",
        "loc_img_x_B": "x_2d",
        "loc_img_y_B": "y_2d",
    }
)

paired_points = pd.concat([paired_points_A, paired_points_B]).drop_duplicates(
    ["camera", "bundle", "id"]
)
# Get points by camera...go back to paired point data

#%%
merged_point_data = (
    paired_points.merge(points_3d_df, how="left", on=["bundle", "id"])
    .sort_values(["bundle", "id"])
    .dropna()
)

#%%
points_3d = np.array(points_3d_df[["x_3d", "y_3d", "z_3d"]])

n_points = points_3d.shape[0]
point_indices = np.array(merged_point_data["index_3d"], dtype=np.int32)
camera_indices = np.array(merged_point_data["camera"], dtype=np.int32)
points_2d = np.array(merged_point_data[["x_2d","x_3d"]])

# camera_id
# camera_id with shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.


# point_ind
# point_ind with shape (n_observations,) contains indices of points (from 0 to n_points - 1) involved in each observation.


# points_2d
# points_2d with shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each observations.

# %%
# points_3d
n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))


# Now define the function which returns a vector of residuals. We use numpy vectorized computations:

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid="ignore"):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return (
        cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
    )


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9 :].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


# You can see that computing Jacobian of fun is cumbersome,
# thus we will rely on the finite difference approximation.
# To make this process time feasible we provide Jacobian
# sparsity structure (i. e. mark elements which are known to be non-zero):


from scipy.sparse import lil_matrix


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


# %matplotlib inline
import matplotlib.pyplot as plt

x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
# plt.plot(f0)

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

import time
from scipy.optimize import least_squares

t0 = time.time()
res = least_squares(
    fun,
    x0,
    jac_sparsity=A,
    verbose=2,      
    x_scale="jac",
    ftol=1e-4,
    method="trf",
    args=(n_cameras, n_points, camera_indices, point_indices, points_2d),
)
t1 = time.time()


# %%
