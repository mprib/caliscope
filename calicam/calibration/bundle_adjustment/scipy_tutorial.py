# Example taken from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
#%%
from __future__ import print_function

import urllib.request
import bz2
import os
import numpy as np
from pathlib import Path

repo = str(Path(__file__)).split("src")[0]
directory = Path(repo, r"src\calibration\bundle_adjustment")

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
URL = BASE_URL + FILE_NAME


if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)

#%%
def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))


        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d


camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(
    FILE_NAME
)


"""
Here we have numpy arrays:

camera_params with shape (n_cameras, 9) contains initial estimates of parameters for all cameras. 
First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), 
next 3 components form a translation vector, then a focal distance and two distortion parameters.

points_3d with shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
camera_id with shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
point_ind with shape (n_observations,) contains indices of points (from 0 to n_points - 1) involved in each observation.
points_2d with shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each observations.
"""

n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))


# Now define the function which returns a vector of residuals. We use numpy vectorized computations:
#%% 

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

