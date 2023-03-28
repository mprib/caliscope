# import saved point data and initial array configuration data
# currently this is for the convenience of not having to rerun everything
# though this workflow may be useful into the future. Save out milestone calculations
# along the way that allow for blocks of dataprocessing
#%%
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from pathlib import Path

from scipy.sparse import lil_matrix

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.calibration.capture_volume.helper_functions.get_stereotriangulated_table import get_stereotriangulated_table

CAMERA_PARAM_COUNT = 6  # this will evolve when moving from extrinsic to intrinsic


@dataclass
class PointEstimates:
    """
    Initialized from triangulated_points.csv to provide the formatting of data required for bundle adjustment
    "full" is used here because there is currently a method to filter the data based on reprojection error
    Not sure if it will be used going forward, but it remains here if so.
    """ 

    sync_indices: np.ndarray  # the sync_index from when the image was taken
    camera_indices: np.ndarray  # camera id associated with the img point
    point_id: np.ndarray # point id (i.e. charuco corner currently)
    img: np.ndarray  # x,y coords of point
    obj_indices: np.ndarray # mapping of x,y img points to their respective list of estimated x,y,z obj points
    obj: np.ndarray  # x,y,z estimates of object points
    # obj_corner_id: np.ndarray # the charuco corner ID of the xyz object point; is this necessary?
    

    @property
    def n_cameras(self):
        return np.unique(self.camera_indices).size

    @property
    def n_obj_points(self):
        return self.obj.shape[0]

    @property
    def n_img_points(self):
        return self.img.shape[0]

    def get_sparsity_pattern(self):
        """provide the sparsity structure for the Jacobian (elements that are not zero)
        n_points: number of unique 3d points; these will each have at least one but potentially more associated 2d points
        point_indices: a vector that maps the 2d points to their associated 3d point
        """

        m = self.camera_indices.size * 2
        n = self.n_cameras * CAMERA_PARAM_COUNT + self.n_obj_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(self.camera_indices.size)
        for s in range(CAMERA_PARAM_COUNT):
            A[2 * i, self.camera_indices * CAMERA_PARAM_COUNT + s] = 1
            A[2 * i + 1, self.camera_indices * CAMERA_PARAM_COUNT + s] = 1

        for s in range(3):
            A[2 * i, self.n_cameras * CAMERA_PARAM_COUNT + self.obj_indices * 3 + s] = 1
            A[
                2 * i + 1,
                self.n_cameras * CAMERA_PARAM_COUNT + self.obj_indices * 3 + s,
            ] = 1

        return A

    def update_obj_xyz(self, least_sq_result_x):
        """
        Provided with the least_squares estimate of the best fit of model parameters (including camera 6DoF)
        parse out the x,y,z object positions and update self.obj
        """
        
        xyz = least_sq_result_x[self.n_cameras * CAMERA_PARAM_COUNT :]
        xyz = xyz.reshape(-1, 3)

        self.obj = xyz
        
        
        
        
        



#%%
if __name__ == "__main__":
    #%%
    from pyxy3d import __root__
    from pyxy3d.cameras.camera_array_builder_deprecate import CameraArrayBuilder
    from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import get_point_estimates 
    session_directory = Path(__root__, "tests", "4_cameras_beginning")
    point_data_path = Path(session_directory, "point_data.csv" )

    camera_array = CameraArrayBuilder(Path(session_directory, "config.toml")).get_camera_array()

    point_estimates = get_point_estimates(camera_array, point_data_path)



# %%
