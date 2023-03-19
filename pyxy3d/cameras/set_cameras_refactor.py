# %%
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)


from pathlib import Path
from pyxy3d.cameras.camera_array_builder import CameraArrayBuilder
from pyxy3d.cameras.camera_array import CameraData, CameraArray
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)

from pyxy3d import __root__
import numpy as np
from dataclasses import dataclass, asdict
import toml



@dataclass
class StereoPair:
    """
    A dataclass to hold the extrinsic parameters associated with the cv2.stereoCalibrate
    function output. Additionally provides some convenience methods to get common transformations
    of the data.

    From the first output of cv2.stereocalibrate, variations can be created by inverting camera
    relationships, and chaining together relative camera positions so that all possible pairs
    of cameras are represented. This dataclass is the building block of that larger process.
    """

    primary_port: int
    secondary_port: int
    error_score: float  # when chaining operations, this will be a cumulative number
    translation: np.ndarray
    rotation: np.ndarray

    @property
    def pair(self):
        return (self.primary_port, self.secondary_port)

    @property
    def transformation(self):

        R_stack = np.vstack([self.rotation, np.array([0, 0, 0])])
        t_stack = np.vstack([self.translation, np.array([1])])
        Tranformation = np.hstack([R_stack, t_stack])
        return Tranformation


class CameraArrayInitializer:
    def __init__(self, config_path: Path):

        logger.info("Creating initial estimate of camera array based on stereopairs...")

        self.config = toml.load(config_path)
        self.ports = self._get_ports()
        self.all_stereopairs = self._get_all_stereopairs()
        # TODO: Need to create the abililty to manufacture "bridged" pairs with
        # cumulative error scores that can be used to initialize an array
        # from any anchor, even with incomplete stereopair coverage..
        self.best_camera_array = self.get_best_camera_array()

    def get_initial_camera_array(self) -> CameraArray:
        return self.best_camera_array

    def _get_ports(self) -> list:
        ports = []
        for key, params in self.config.items():
            if key.split("_")[0] == "stereo":
                port_A = int(key.split("_")[1])
                port_B = int(key.split("_")[2])
                ports.append(port_A)
                ports.append(port_B)

        # convert to a unique list
        ports = list(set(ports))
        return ports

    def _get_all_stereopairs(self) -> dict:

        stereopairs = {}

        # Create StereoPair objects for each saved stereocalibration output in config
        # this are maintained in a dictionary keyed off of the pair tuple
        for key, params in self.config.items():
            if key.split("_")[0] == "stereo":
                port_A = int(key.split("_")[1])
                port_B = int(key.split("_")[2])

                rotation = np.array(params["rotation"], dtype=np.float64)
                translation = np.array(params["translation"], dtype=np.float64)
                error = float(params["RMSE"])

                new_stereopair = StereoPair(
                    primary_port=port_A,
                    secondary_port=port_B,
                    error_score=error,
                    translation=translation,
                    rotation=rotation,
                )

                stereopairs[new_stereopair.pair] = new_stereopair

        # create another dictionary that will contain the inverted versions of the StereoPairs
        inverted_stereopairs = {}
        for pair, stereopair in stereopairs.items():
            a, b = pair
            inverted_pair = (b, a)
            inverted_stereopairs[inverted_pair] = get_inverted_stereopair(stereopair)

        # combine the dictionaries
        merged_stereopairs = {**stereopairs, **inverted_stereopairs}
        return merged_stereopairs

    def _get_scored_anchored_array(self, anchor_port: int) -> tuple:
        """
        Constructs a complete camera array based on the available stereopairs in
        self.all_stereopairs

        two return values:

            total_error_score: the sum of the error_scores of all stereopairs used in the
                            construction of the array

            camera_array: a CameraArray object anchored at the provided port
        """
        cameras = {}
        total_error_score = 0

        for key, data in self.config.items():
            if key.startswith("cam_") and not self.config[key]["ignore"]:
                port = data["port"]
                size = data["size"]
                rotation_count = data["rotation_count"]
                error = data["error"]
                matrix = np.array(data["matrix"], dtype=np.float64)
                distortions = np.array(data["distortions"], dtype=np.float64)
                exposure = data["exposure"]
                grid_count = data["grid_count"]
                ignore = data["ignore"]
                verified_resolutions = data["verified_resolutions"]

                # update with extrinsics, though place anchor camera at origin
                if port == anchor_port:
                    translation = np.array([0, 0, 0], dtype=np.float64).T
                    rotation = np.array(
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
                    )
                else:
                    anchored_stereopair = self.all_stereopairs[(anchor_port, port)]
                    translation = anchored_stereopair.translation[:, 0]
                    rotation = anchored_stereopair.rotation
                    total_error_score += anchored_stereopair.error_score

                cam_data = CameraData(
                    port,
                    size,
                    rotation_count,
                    error,
                    matrix,
                    distortions,
                    exposure,
                    grid_count,
                    ignore,
                    verified_resolutions,
                    translation,
                    rotation,
                )

                cameras[port] = cam_data

        camera_array = CameraArray(cameras)

        return total_error_score, camera_array

    def get_best_camera_array(self):
        """
        returns the anchored camera array with the lowest total error score.
        Note that total error score is just a sum of individual errors for tracking
        and comparison purposes and does not have any signifigence in the context
        of reprojection error

        """
        array_error_scores = {}
        camera_arrays = {}
        # get the score for the anchored_stereopairs

        # TODO: ned to set ports elsewhere in code
        for port in self.ports:
            array_error_score, camera_array = self._get_scored_anchored_array(port)
            array_error_scores[port] = array_error_score
            camera_arrays[port] = camera_array

        best_anchor = min(array_error_scores, key=array_error_scores.get)

        best_initial_array = camera_arrays[best_anchor]

        return best_initial_array


def get_inverted_stereopair(stereo_pair: StereoPair) -> StereoPair:
    primary_port = stereo_pair.secondary_port
    secondary_port = stereo_pair.primary_port
    error_score = stereo_pair.error_score

    inverted_transformation = np.linalg.inv(stereo_pair.transformation)
    rotation = inverted_transformation[0:3, 0:3]
    translation = inverted_transformation[0:3, 3:]

    inverted_stereopair = StereoPair(
        primary_port=primary_port,
        secondary_port=secondary_port,
        error_score=error_score,
        translation=translation,
        rotation=rotation,
    )
    return inverted_stereopair


# def get_anchored_pairs(anchor: int, all_stereopairs:dict)->dict:


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    from pyxy3d.gui.vizualize.capture_volume_visualizer import CaptureVolumeVisualizer
    from pyxy3d.session import Session

    # session_directory = Path(__root__, "tests", "3_cameras_middle")
    session_directory = Path(__root__,"tests", "3_cameras_triangular" )
    # session_directory = Path(__root__,"tests", "3_cameras_midlinear" )

    session = Session(session_directory)
    config_path = Path(session_directory, "config.toml")

    camera_array = CameraArrayInitializer(config_path).get_best_camera_array()

    point_data_path = Path(session_directory, "point_data.csv")

    point_estimates: PointEstimates = get_point_estimates(camera_array, point_data_path)

    capture_volume = CaptureVolume(camera_array, point_estimates)

    capture_volume.save(session_directory)
    #%%

    capture_volume.optimize()
    capture_volume.save(session_directory)
    #%%
    app = QApplication(sys.argv)
    vizr = CaptureVolumeVisualizer(capture_volume=capture_volume)
    sys.exit(app.exec())


#%%
# this is an awesome two-liner to convert a dictionary of dataclasses to a pandas dataframe
# stereopair_dict = {k:asdict(merged_stereopairs) for k,merged_stereopairs in merged_stereopairs.items()}
# df = pd.DataFrame(list(stereopair_dict.values()), index=stereopair_dict.keys())


# %%
