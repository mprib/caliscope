#%%

import time
from scipy.optimize import least_squares
from pathlib import Path

import sys
repo = str(Path(__file__)).split("src")[0]
sys.path.insert(0,repo)

from src.calibration.bundle_adjustment.bundle_adjust_functions import *

session_directory = Path(repo, "sessions", "iterative_adjustment")

config_path = Path(session_directory, "config.toml")
array_builder = CameraArrayBuilder(config_path)
camera_array = array_builder.get_camera_array()

camera_params = get_camera_params(camera_array)


points_csv_path = Path(session_directory, "recording", "triangulated_points.csv")
points_2d_df = get_points_2d_df(points_csv_path)
points_3d_df = get_points_3d_df(points_csv_path)

    
camera_indices, point_indices, points_2d, points_3d, n_points = get_bundle_adjust_params(points_2d_df, points_3d_df)
    
    
#%%
    
n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))
    
x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)


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

res.x
    
flat_camera_params = res.x[0 : n_cameras * 9]
n_params = 9
new_camera_params = flat_camera_params.reshape(n_cameras, n_params)
print(new_camera_params)
# %%

for port, cam i


