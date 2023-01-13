#%%
import time
from pathlib import Path
import pickle
import sys

repo = str(Path(__file__)).split("src")[0]
sys.path.insert(0, repo)
print(repo)

#%%
from src.calibration.bundle_adjustment.bundle_adjust_functions import *

#%%
session_directory = Path(repo, "sessions", "iterative_adjustment")

config_path = Path(session_directory, "config.toml")
array_builder = CameraArrayBuilder(config_path)
camera_array = array_builder.get_camera_array()


points_csv_path = Path(session_directory, "recording", "triangulated_points.csv")

res = bundle_adjust(camera_array, points_csv_path)

res_path = Path(session_directory, "res.pkl")
# with open(res_path, "wb") as file:
# pickle.dump(res, file)

with open(res_path, "rb") as file:
    res = pickle.load(file)

#%%
# print(res)

n_cameras = len(camera_array.cameras)
flat_camera_params = res.x[0 : n_cameras * 9]
n_params = 9
new_camera_params = flat_camera_params.reshape(n_cameras, n_params)
# print(new_camera_params)

# update camera array with new positional data
for index in range(len(new_camera_params)):
    print(index)
    cam_vec = new_camera_params[index, :]
    camera_array.cameras[index].from_vector(cam_vec)


from src.gui.capture_volume.visualizer import CaptureVolumeVisualizer

from PyQt6.QtWidgets import QApplication

app = QApplication(sys.argv)
vizr = CaptureVolumeVisualizer(camera_array)
sys.exit(app.exec())
# %%
