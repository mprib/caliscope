#%%
import calicam.logger

logger = calicam.logger.get(__name__)

from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
import toml
from calicam.calibration.charuco import Charuco
from calicam.calibration.capture_volume.capture_volume import (
    CaptureVolume,
    xy_reprojection_error,
)


class QualityScanner:
    def __init__(self, session_directory: Path, capture_volume_path: Path):
        self.session_directory = session_directory
        self.config_path = Path(self.session_directory, "config.toml")

        self.charuco = self.get_charuco()
        self.capture_volume = self.get_capture_volume(capture_volume_path)

    def get_charuco(self) -> Charuco:
        config = toml.load(self.config_path)

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

    def get_capture_volume(self, capture_volume_pkl_path: Path) -> CaptureVolume:
        with open(capture_volume_pkl_path, "rb") as file:
            capture_volume = pickle.load(file)
        return capture_volume

    def get_summary_df(self) -> pd.DataFrame:
        """
        Unpack the Array Diagnostic data into a pandas dataframe format that can be
        plotted and summarized. This is an omnibus dataframe that can be inspected for
        accuracy and form the basis of additional slices used for a given purpose.

        In particular, this data will get reformatted and paired down to create the inputs
        used for the Charuco corner distance calculation.
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
        return summarized_data


#%%
if __name__ == "__main__":
    #%%
    
    