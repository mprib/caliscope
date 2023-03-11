#%%
import calicam.logger

logger = calicam.logger.get(__name__)

from pathlib import Path
import pickle
import sys
from scipy import stats
import numpy as np
import pandas as pd
import toml
from calicam.calibration.charuco import Charuco
from calicam.calibration.capture_volume.capture_volume import (
    CaptureVolume,
    xy_reprojection_error,
)

from calicam.calibration.capture_volume.point_estimates import PointEstimates

class QualityController:
    def __init__(self, capture_volume:CaptureVolume, charuco:Charuco):
        # self.session_directory = session_directory
        # self.config_path = Path(self.session_directory, "config.toml")

        # pull charuco from config
        self.charuco = charuco

        # load capture volume being analyzed
        # capture_volume_path = Path(self.session_directory, capture_volume_name)

        self.capture_volume = capture_volume

        # all 2d data including reprojection error and estimated corresponding 3d point
        self.data_2d = self.get_summary_2d_df()

        # all individual 3d points estimated in a world frame of reference
        self.corners_world_xyz = self.get_corners_world_xyz()

        # all possible pairs of 3d points that share the same sync_index
        # these align with the index used in self.corners_xyz
        self.paired_obj_indices = self.get_paired_obj_indices()

        # Corner positions in a board frame of refernce aligning with index of corners_world_xyz
        # note this is already just a numpy ndarray
        self.corners_board_xyz = self.get_corners_board_xyz()

        self.distance_error = self.get_distance_error()



    def get_summary_2d_df(self) -> pd.DataFrame:
        """
        Unpack the Array Diagnostic data into a pandas dataframe format that can be
        plotted and summarized. This is all 2d data observations with their
        corresponding 3d point estimates (meaning the 3d point data is duplicated)
        """

        capture_volume_xy_error = xy_reprojection_error(
            self.capture_volume.get_vectorized_params(), self.capture_volume
        ).reshape(-1, 2)
        # build out error as singular distanc

        xyz = self.capture_volume.get_xyz_points()

        euclidean_distance_error = np.sqrt(np.sum(capture_volume_xy_error**2, axis=1))
        row_count = euclidean_distance_error.shape[0]

        array_data_dict = {
            "camera": self.capture_volume.point_estimates.camera_indices.tolist(),
            "sync_index": self.capture_volume.point_estimates.sync_indices.astype(
                int
            ).tolist(),
            "charuco_id": self.capture_volume.point_estimates.point_id.tolist(),
            "img_x": self.capture_volume.point_estimates.img[:, 0].tolist(),
            "img_y": self.capture_volume.point_estimates.img[:, 1].tolist(),
            "reproj_error_x": capture_volume_xy_error[:, 0].tolist(),
            "reproj_error_y": capture_volume_xy_error[:, 1].tolist(),
            "reproj_error": euclidean_distance_error.tolist(),
            "reproj_error_sq": (euclidean_distance_error**2).tolist(),
            "obj_id": self.capture_volume.point_estimates.obj_indices.tolist(),
            "obj_x": xyz[self.capture_volume.point_estimates.obj_indices][
                :, 0
            ].tolist(),
            "obj_y": xyz[self.capture_volume.point_estimates.obj_indices][
                :, 1
            ].tolist(),
            "obj_z": xyz[self.capture_volume.point_estimates.obj_indices][
                :, 2
            ].tolist(),
        }

        summarized_data = pd.DataFrame(array_data_dict).astype(
            {"sync_index": "int32", "charuco_id": "int32", "obj_id": "int32"}
        )

        summarized_data["reproj_error_percentile"] = stats.percentileofscore(
            summarized_data["reproj_error"], summarized_data["reproj_error"]
        )

        return summarized_data

    def get_corners_world_xyz(self) -> pd.DataFrame:
        """
        convert the table of 2d data observations to a smaller table of only the individual 3d point
        estimates. These will have a number of duplicates so drop them.
        """

        corners_3d = (
            self.data_2d[
                ["sync_index", "charuco_id", "obj_id", "obj_x", "obj_y", "obj_z"]
            ]
            .astype({"sync_index": "int32", "charuco_id": "int32", "obj_id": "int32"})
            .drop_duplicates()
            .sort_values(by=["obj_id"])
            .reset_index()
        )

        return corners_3d

    def get_paired_obj_indices(self) -> np.ndarray:
        """given a dataframe that contains all observed charuco corners across sync_indices,
        return a Nx2 matrix of paired object indices that will represent all possible
        joined lines between charuco corners for each sync_index"""

        # get columns out from data frame for numpy calculations
        sync_indices = self.corners_world_xyz["sync_index"].to_numpy(dtype=np.int32)
        unique_sync_indices = np.unique(sync_indices)
        obj_id = self.corners_world_xyz["obj_id"].to_numpy(dtype=np.int32)

        # for a given sync index (i.e. one board snapshot) get all pairs of object ids
        paired_obj_indices = None
        for x in unique_sync_indices:
            sync_obj = obj_id[
                sync_indices == x
            ]  # 3d objects (corners) at a specific sync_index
            all_pairs = cartesian_product(sync_obj, sync_obj)
            if paired_obj_indices is None:
                paired_obj_indices = all_pairs
            else:
                paired_obj_indices = np.vstack([paired_obj_indices, all_pairs])

        # paired_corner_indices will contain duplicates (i.e. [0,1] and [1,0]) as well as self-pairs ([0,0], [1,1])
        # this need to get filtered out
        reformatted_paired_obj_indices = np.zeros(
            paired_obj_indices.shape, dtype=np.int32
        )
        reformatted_paired_obj_indices[:, 0] = np.min(
            paired_obj_indices, axis=1
        )  # smaller on left
        reformatted_paired_obj_indices[:, 1] = np.max(
            paired_obj_indices, axis=1
        )  # larger on right
        reformatted_paired_obj_indices = np.unique(
            reformatted_paired_obj_indices, axis=0
        )
        reformatted_paired_obj_indices = reformatted_paired_obj_indices[
            reformatted_paired_obj_indices[:, 0] != reformatted_paired_obj_indices[:, 1]
        ]

        return reformatted_paired_obj_indices

    def get_corners_board_xyz(self) -> np.ndarray:
        corner_ids = self.corners_world_xyz["charuco_id"]
        corners_board_xyz = self.charuco.board.chessboardCorners[corner_ids]

        return corners_board_xyz

    def get_distance_error(self) -> pd.DataFrame:

        # temp numpy frame for working calculations
        corners_world_xyz = self.corners_world_xyz[
            ["obj_x", "obj_y", "obj_z"]
        ].to_numpy()
        corners_board_xyz = self.corners_board_xyz

        # get the xyz positions for all pairs of corners
        corners_world_A = corners_world_xyz[self.paired_obj_indices[:, 0]]
        corners_world_B = corners_world_xyz[self.paired_obj_indices[:, 1]]
        corners_board_A = corners_board_xyz[self.paired_obj_indices[:, 0]]
        corners_board_B = corners_board_xyz[self.paired_obj_indices[:, 1]]

        # get the distance between them
        distance_world_A_B = np.sqrt(
            np.sum((corners_world_A - corners_world_B) ** 2, axis=1)
        )
        distance_board_A_B = np.sqrt(
            np.sum((corners_board_A - corners_board_B) ** 2, axis=1)
        )

        distance_world_A_B = np.round(distance_world_A_B, 5)
        distance_board_A_B = np.round(distance_board_A_B, 5)

        # calculate error (in mm)
        distance_error = distance_world_A_B - distance_board_A_B
        
        # wrap everything up in a dataframe for ease of processing
        distance_error = pd.DataFrame(distance_error, columns=["Distance_Error"])
        distance_error["Distance_Error_mm"] = distance_error["Distance_Error"] * 1000
        distance_error["Distance_Error_mm_abs"] = abs(
            distance_error["Distance_Error_mm"]
        )

        distance_error["corner_A"] = self.paired_obj_indices[:,0]
        distance_error["corner_B"] = self.paired_obj_indices[:,1]

        distance_error["world_distance"] = distance_world_A_B       
        distance_error["board_distance"] = distance_board_A_B       

        return distance_error

    def get_filtered_data_2d(self, percentile_cutoff: float):
        """
        Provided a cutoff percentile value, returns a filtered_data_2d dataframe
        that only represents observations that have a reprojection error below
        that threshold. Additionally, it removes any singular 2d observations
        (i.e. those that have only one snapshot image and therefore cannot
        be localized in 3d)

        percentile_cutoff: a fraction between 0 and 1
        """

        # filter data based on reprojection error
        filtered_data_2d = self.data_2d.query(
            f"reproj_error_percentile <{str(percentile_cutoff*100)}"
        )

        # get the count of obj_ids to understand how many times
        # each 3d object is represented in the 2d data
        obj_id_counts = (
            filtered_data_2d.filter(["obj_id", "camera"])
            .groupby("obj_id")
            .count()
            .rename(columns={"camera": "obj_id_count"})
        )

        # merge back into the filtered data
        filtered_data_2d = filtered_data_2d.merge(obj_id_counts, "right", on=["obj_id"])

        # remove any points that now only have 1 2d image associated with them
        filtered_data_2d = filtered_data_2d.query("obj_id_count > 1").rename(
            columns={"obj_id": "original_obj_id"}
        )
        return filtered_data_2d

def get_capture_volume(capture_volume_pkl_path: Path) -> CaptureVolume:
    with open(capture_volume_pkl_path, "rb") as file:
        capture_volume = pickle.load(file)
    return capture_volume

def get_charuco(config_path) -> Charuco:
    config = toml.load(config_path)

    ## create charuco
    charuco = Charuco(
        columns=config["charuco"]["columns"],
        rows=config["charuco"]["rows"],
        board_height=config["charuco"]["board_height"],
        board_width=config["charuco"]["rows"],
        dictionary=config["charuco"]["dictionary"],
        units=config["charuco"]["units"],
        aruco_scale=config["charuco"]["aruco_scale"],
        square_size_overide_cm=config["charuco"]["square_size_overide_cm"],
        inverted=config["charuco"]["inverted"],
    )

    return charuco

def cartesian_product(*arrays):
    """
    helper function for creating all possible pairs of points within a given sync_index
    https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


if __name__ == "__main__":
# if True:
    from calicam import __root__

    session_directory = Path(__root__, "tests", "demo")
    config_path = Path(session_directory, "config.toml")  

    capture_volume_name = "capture_volume_stage_1.pkl"
    
    charuco = get_charuco(config_path)
    cap_vol_1 = get_capture_volume(Path(session_directory,capture_volume_name))

    quality_filter = QualityController(cap_vol_1,charuco)
    data_2d = quality_filter.data_2d

    data_2d.to_csv(Path(session_directory, "data_2d.csv"))

    # corners_world_xyz = q_f_1.corners_world_xyz
    # paired_indices = q_f_1.paired_obj_indices
    distance_error = quality_filter.distance_error

    distance_error.to_csv(Path(session_directory,"distance_error.csv"))

    logger.info(distance_error.describe())


    percentile_cutoff = 0.5

    filtered_data_2d = quality_filter.get_filtered_data_2d(percentile_cutoff)

    objects_3d = (
        filtered_data_2d.filter(["original_obj_id", "obj_x", "obj_y", "obj_z"])
        .drop_duplicates()
        .reset_index()
        .drop("index", axis=1)
        .reset_index()
        .rename(columns={"index":"filtered_obj_id"})
    )
    
    old_new_mapping = objects_3d.filter(["filtered_obj_id", "original_obj_id"])
    
    filtered_data_2d = filtered_data_2d.merge(old_new_mapping, how="right", on=["original_obj_id"])


    # get revised point_estimates
    sync_indices = filtered_data_2d["sync_index"].to_numpy()
    camera_indices = filtered_data_2d["camera"].to_numpy()
    point_id = filtered_data_2d["charuco_id"].to_numpy()
    img = filtered_data_2d.filter(["img_x", "img_y"]).to_numpy()
    obj_indices = filtered_data_2d["filtered_obj_id"].to_numpy()
    obj = objects_3d.filter(["obj_x", "obj_y", "obj_z"]).to_numpy()

    filtered_point_estimates = PointEstimates(
        sync_indices=sync_indices,
        camera_indices=camera_indices,
        point_id=point_id,
        img=img,
        obj_indices=obj_indices,
        obj=obj
    )
    
    
    quality_filter.capture_volume.point_estimates = filtered_point_estimates
    
    test_filter_directory = Path(__root__, "tests", "demo", "test_filter")
    quality_filter.capture_volume.optimize()
    capture_volume_name = "post_optimized_capture_volume.pkl"
    post_filter_q_s = QualityController(test_filter_directory,capture_volume_name)
    
    
    logger.info("Examinging reprojection error...should reduce")
    logger.info("Pre Filter:")
    logger.info(quality_filter.data_2d["reproj_error"].describe())
    logger.info(quality_filter.get_distance_error().describe())

    logger.info("Post Filter:")
    logger.info(post_filter_q_s.get_distance_error().describe())
    
    logger.info(post_filter_q_s.data_2d["reproj_error"].describe())
    
# %%
