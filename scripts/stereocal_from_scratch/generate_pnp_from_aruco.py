"""
The intention of this branch is to create a pipeline that generates
pairwise camera pose estimates using PnP rather than stereocalibrate.

We have charuco data to perform the chAruco based calculations in parallel.
These will be used as a check for quality.

As an additional step, the pairwise estimates can be refined via bundle adjustment.

"""

import logging
import pandas as pd


from caliscope.logger import setup_logging
from generate_keypoints_from_tracker import generate_aruco_xy
from caliscope import __root__
from caliscope.configurator import Configurator
from caliscope.calibration.stereocalibrator import StereoCalibrator
from caliscope.cameras.camera_array import CameraArray

setup_logging()

logger = logging.getLogger(__name__)


# Use post_optimization test data (has calibration videos)
# these are used in the function above that generates aruco_xy
test_data_dir = __root__ / "tests/sessions/post_optimization"
project_fixture_dir = __root__ / "scripts/fixtures/aruco_pipeline"
calibration_video_dir = project_fixture_dir / "calibration/extrinsic"

stages_to_run = [2]


# load in basic project setup data.
# Note that intrinsic params for cameras should be available
charuco_points_file = calibration_video_dir / "CHARUCO/xy_CHARUCO.csv"
config = Configurator(project_fixture_dir)
camera_array: CameraArray = config.get_camera_array()

# Stage 1: create gold standard reference output.
# relative stereopair poses created by conventional call to cv2.stereocalibrate with charuco calibration boards

if 1 in stages_to_run:
    stereocal = StereoCalibrator(camera_array=camera_array, point_data_path=charuco_points_file)
    stereopose_gold_standard: dict = stereocal.stereo_calibrate_all()

    logger.info("=" * 20)
    logger.info(f"Stereopair relative poses: {stereopose_gold_standard}")
    logger.info("=" * 20)

# It is occuring to me now that the primary aim at the outset is to estimate the stereopair poses from the charuco

# Stage 2: Estimate stereoposes using charuco board and pnp
if 2 in stages_to_run:
    # import raw point data as dataframe
    charuco_points = pd.read_csv(charuco_points_file)

    # log basic data structure for context
    logger.info("+" * 20)

# Stage 3: estiamte stereoposes using only single aruco and pnp
## Generate aruco point data
if "3a" in stages_to_run:
    generate_aruco_xy()

if "3b" in stages_to_run:
    # load in aruco data
    aruco_points_file = calibration_video_dir / "ARUCO/xy_ARUCO.csv"
    aruco_points = pd.read_csv(aruco_points_file)
