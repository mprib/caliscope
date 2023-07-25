
# %%
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
point_estimates_dict = asdict(point_estimates)

for key, params in point_estimates_dict.items():
    point_estimates_dict[key] = params.tolist()


def shape(lst):
    try:
        return [len(lst)] + shape(lst[0])
    except TypeError:
        return []
 

for key, value in point_estimates_dict.items():
    print(key, shape(value))
# point_estimates_path = Path(test_session_path, "point_estimates.csv")

# pd.DataFrame(flat_dict).to_csv(point_estimates_path)
# %%
