#%%
from pathlib import Path

from calicam.session import Session
from calicam import __root__
from calicam.calibration.omnicalibrator import OmniCalibrator
from calicam.calibration.capture_volume.point_estimates import PointEstimates

from calicam.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)


# the session used for the single camera calibration and 
# the omniframe data collection
session_directory = Path(__root__,"tests", "mimic_anipose" )


# point_data.csv is created during the omniframe datacollection
point_data_path = Path(session_directory, "point_data.csv")

# where single camera data is pulled from annd
# stereopair data is updated to
config_path = Path(session_directory, "config.toml")

# with the point data, the stereocalibrations can be performed.
# Note that this is named OmniCalibrator because it has a poorly
# working single camera calibration method as well, but I wouldn't recommend
# might be better termed stereocalibrator going forward
omnicalibrator = OmniCalibrator(config_path, point_data_path)

# create the pairwise estimates of camera positions
# this will save out "stereo_A_B" data to the config.toml
omnicalibrator.stereo_calibrate_all()

# the pairwise estimates will be used to create the initial
# estimate of the camera array positions (now in the config file)
session = Session(session_directory)
session.load_camera_array()

# The 3D point estimates are constructed from the estimated camera 
# array. Stereopair triangulations are made and 
# averaged together for each point. 
# This is used to initialize the bundle adjustment
point_estimates: PointEstimates = get_point_estimates(
    session.camera_array, point_data_path
)

#%%

session.save_camera_array()

#%%



