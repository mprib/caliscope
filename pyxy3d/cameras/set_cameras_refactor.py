#%%
    
from pathlib import Path
from pyxy3d.cameras.camera_array_builder import CameraArrayBuilder    
from pyxy3d import __root__
import numpy as np

 
session_directory = Path(__root__,  "tests", "3_cameras_middle")
config_path = Path(session_directory, "config.toml")


camera_array_builder = CameraArrayBuilder(config_path)
extrinsics = camera_array_builder.get_default_stereoextrinsics()
#%%
test_transform = extrinsics[(0,2)]["Transformation"]

# %%
