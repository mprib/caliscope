

import calicam.logger
logger = calicam.logger.get(__name__)

from pathlib import Path
import pickle
from dataclasses import dataclass
import numpy as np


from calicam.calibration.capture_volume.point_estimate_data import PointEstimateData
from calicam.cameras.camera_array import CameraArray

CAMERA_PARAM_COUNT = 6
@dataclass
class CaptureVolume:
    point_estimate_data: PointEstimateData
    model_params: np.ndarray  # the first argument of the residual function
    xy_reprojection_error: np.ndarray
    camera_array: CameraArray

    def save(self, output_path):
        with open(Path(output_path), "wb") as file:
            pickle.dump(self, file)

    
    def get_xyz_points(self):
        """Get 3d positions arrived at by bundle adjustment"""
        n_cameras = len(self.camera_array.cameras)
        xyz = self.model_params[n_cameras * CAMERA_PARAM_COUNT :]
        xyz = xyz.reshape(-1, 3)

        return xyz
    
    def get_summary_df(self, label:str):
        
        array_data_xy_error = self.xy_reprojection_error.reshape(-1, 2)
        # build out error as singular distance

        xyz = self.get_xyz_points()

        euclidean_distance_error = np.sqrt(np.sum(array_data_xy_error**2, axis=1))
        row_count = euclidean_distance_error.shape[0]

        array_data_dict = {
            "label": [label] * row_count,
            "camera": self.point_estimate_data.camera_indices_full.tolist(),
            "sync_index": self.point_estimate_data.sync_indices.astype(int).tolist(),
            "charuco_id": self.point_estimate_data.corner_id.tolist(),
            "img_x": self.point_estimate_data.img_full[:, 0].tolist(),
            "img_y": self.point_estimate_data.img_full[:, 1].tolist(),
            "reproj_error_x": array_data_xy_error[:, 0].tolist(),
            "reproj_error_y": array_data_xy_error[:, 1].tolist(),
            "reproj_error": euclidean_distance_error.tolist(),
            "obj_id": self.point_estimate_data.obj_indices.tolist(),
            "obj_x": xyz[self.point_estimate_data.obj_indices_full][:, 0].tolist(),
            "obj_y": xyz[self.point_estimate_data.obj_indices_full][:, 1].tolist(),
            "obj_z": xyz[self.point_estimate_data.obj_indices_full][:, 2].tolist(),
        }

        summarized_data = (pd.DataFrame(array_data_dict)
                            .astype({"sync_index":'int32', "charuco_id":"int32", "obj_id":"int32"})
        )
        return summarized_data