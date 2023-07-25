#%%


import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from pathlib import Path
import pandas as pd
from pyxy3d.configurator import Configurator
from pyxy3d import __root__
from pyxy3d.cameras.camera_array import CameraArray, CalibrationStage
from dataclasses import asdict

test_session_path = Path(__root__, "tests", "sessions","post_optimization")

config = Configurator(test_session_path)

point_estimates = config.get_point_estimates()
point_estimates_dict = point_estimates.to_flat_dict()
#%%



flat_dict = {}
for key, params in point_estimates_dict.items():
    match key:
        case "img":
            xy = params
            flat_dict["img_x"] = xy[:,0].tolist()
            flat_dict["img_y"] = xy[:,1].tolist()
        case "obj":
            xyz = params
            flat_dict["obj_x"] = xyz[:,0].tolist()
            flat_dict["obj_y"] = xyz[:,1].tolist()
            flat_dict["obj_z"] = xyz[:,2].tolist()
        case _:
            flat_dict[key] = params.tolist() 


for key, value in flat_dict.items():
    print(key, len(value))
# point_estimates_path = Path(test_session_path, "point_estimates.csv")

# pd.DataFrame(flat_dict).to_csv(point_estimates_path)
# %%
