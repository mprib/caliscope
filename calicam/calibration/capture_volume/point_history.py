# import saved point data and initial array configuration data
# currently this is for the convenience of not having to rerun everything
# though this workflow may be useful into the future. Save out milestone calculations
# along the way that allow for blocks of dataprocessing
#%%
import calicam.logger
logger = calicam.logger.get(__name__)

from pathlib import Path

from scipy.sparse import lil_matrix

import pandas as pd
import numpy as np
from dataclasses import dataclass

CAMERA_PARAM_COUNT = 6  # this will evolve when moving from extrinsic to intrinsic


@dataclass
class PointHistory:
    """
    Initialized from triangulated_points.csv to provide the formatting of data required for bundle adjustment
    "full" is used here because there is currently a method to filter the data based on reprojection error
    Not sure if it will be used going forward, but it remains here if so.
    """ 

    camera_indices: np.ndarray  # camera id associated with the img point
    img: np.ndarray  # x,y coords of point
    point_id: np.ndarray # point id (i.e. charuco corner currently)
    obj_indices: np.ndarray # mapping of x,y img points to their respective list of estimated x,y,z obj points
    obj: np.ndarray  # x,y,z estimates of object points
    obj_corner_id: np.ndarray # the charuco corner ID of the xyz object point; is this necessary?
    sync_indices: np.ndarray  # the sync_index from when the image was taken
    

    def filter(self, least_squares_result_fun, percent_cutoff):
        # I believe this was indentended for use with some iterative approach to bundle adjustment
        # that skimmed off the poor fits and reran, akin to anipose. 
        # it may still be a useful tool...

        xy_reproj_error = least_squares_result_fun.reshape(-1, 2)
        euclidean_distance_error = np.sqrt(np.sum(xy_reproj_error**2, axis=1))

        error_rank = np.argsort(euclidean_distance_error)
        n_2d_points = error_rank.shape[0]
        error_percent_rank = error_rank / n_2d_points

        include = error_percent_rank < percent_cutoff

        full_count = include.size
        subset_count = include[include == True].size

        print(
            f"Reducing point data to {subset_count} image points (full count: {full_count})"
        )

        self.camera_indices = self.camera_indices[include]
        self.obj_indices = self.obj_indices[include]
        self.point_id = self.point_id[include]
        self.img = self.img[include]
        self.sync_indices = self.sync_indices[include]

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
        
        
        
        
        
def get_points_2d_df(points_csv_path):
    points_df = pd.read_csv(points_csv_path)

    points_2d_port_A = points_df[
        ["port_A", "sync_index", "id", "x_A_raw", "y_A_raw"]
    ].rename(
        columns={
            "port_A": "camera",
            "sync_index": "sync_index",
            "id": "corner_id",
            "x_A_raw": "x_2d",
            "y_A_raw": "y_2d",
        }
    )

    points_2d_port_B = points_df[
        ["port_B", "sync_index", "id", "x_B_raw", "y_B_raw"]
    ].rename(
        columns={
            "port_B": "camera",
            "sync_index": "sync_index",
            "id": "corner_id",
            "x_B_raw": "x_2d",
            "y_B_raw": "y_2d",
        }
    )

    points_2d_df = (
        pd.concat([points_2d_port_A, points_2d_port_B])
        .drop_duplicates()
        .sort_values(["sync_index", "corner_id", "camera"])
        .rename(columns={"sync_index": "sync_index"})
    )
    return points_2d_df


# get 3d points with indices to merge back into points_2d
def get_points_3d_df(points_csv_path):
    points_df = pd.read_csv(points_csv_path)
    points_3d_df = (
        points_df[["sync_index", "id", "pair", "x_pos", "y_pos", "z_pos"]]
        .sort_values(["sync_index", "id"])
        .groupby(["sync_index", "id"])
        .agg({"x_pos": "mean", "y_pos": "mean", "z_pos": "mean", "pair": "size"})
        .rename(
            columns={"pair": "count", "x_pos": "x_3d", "y_pos": "y_3d", "z_pos": "z_3d"}
        )
        .reset_index()
        .reset_index()
        .rename(columns={"index": "index_3d", "id": "corner_id"})
    )
    return points_3d_df


def get_merged_2d_3d(points_csv_path):
    """
    For each 2d point line, add in the estimated 3d point position
    """
    points_2d_df = get_points_2d_df(points_csv_path)
    points_3d_df = get_points_3d_df(points_csv_path)

    merged_point_data = (
        points_2d_df.merge(points_3d_df, how="left", on=["sync_index", "corner_id"])
        .sort_values(["camera", "sync_index", "corner_id"])
        .dropna()
    )

    return merged_point_data


def get_point_history(stereo_points_csv_path: Path) -> PointHistory:
    """
    formats the triangulated_points.csv file into a PointEstimateData that has the 
    data structured in a way that is amenable to bundle adjustment
    """
    #NOTE: Not a method of the dataclass, the is a convenience constructor
    points_3d_df = get_points_3d_df(stereo_points_csv_path)
    merged_point_data = get_merged_2d_3d(stereo_points_csv_path)

    camera_indices = np.array(merged_point_data["camera"], dtype=np.int64)
    img = np.array(merged_point_data[["x_2d", "y_2d"]])
    corner_id = np.array(merged_point_data["corner_id"], dtype=np.int64)
    obj_indices = np.array(merged_point_data["index_3d"], dtype=np.int64)
    sync_index = np.array(merged_point_data["sync_index"], dtype=np.int64)
    obj = np.array(points_3d_df[["x_3d", "y_3d", "z_3d"]])
    obj_corner_id = np.array(points_3d_df[["corner_id"]])

    return PointHistory(
        camera_indices=camera_indices,
        img=img,
        point_id=corner_id,
        obj_indices=obj_indices,
        obj=obj,
        obj_corner_id=obj_corner_id,
        sync_indices=sync_index,
    )

#%%
if __name__ == "__main__":
    #%%
    from calicam import __root__
    
    session_directory = Path(__root__, "tests", "5_cameras")
    stereo_points_csv_path = Path(
        session_directory, "recording", "stereotriangulated_points.csv"
    )

    point_data = get_point_history(stereo_points_csv_path)


# %%
