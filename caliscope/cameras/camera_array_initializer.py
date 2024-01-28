# %%


from pathlib import Path
from caliscope.cameras.camera_array import CameraData, CameraArray
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)

from itertools import permutations
from caliscope import __root__
import numpy as np
from dataclasses import dataclass, asdict
import rtoml
import caliscope.logger
logger = caliscope.logger.get(__name__)


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

def get_bridged_stereopair(
    pair_A_B: StereoPair, pair_B_C: StereoPair
) -> StereoPair:
    port_A = pair_A_B.primary_port
    port_C = pair_B_C.secondary_port

    A_B_error = pair_A_B.error_score
    B_C_error = pair_B_C.error_score
    A_C_error = A_B_error + B_C_error

    # new transformations are added on the left
    # https://youtube.com/watch?v=q0mRtuiKSKg&feature=shares&t=66
    bridged_transformation = np.matmul(
        pair_B_C.transformation, pair_A_B.transformation
    )
    bridged_rotation = bridged_transformation[0:3, 0:3]
    bridged_translation = bridged_transformation[None, 0:3, 3].T

    stereo_A_C = StereoPair(
        primary_port=port_A,
        secondary_port=port_C,
        error_score=A_C_error,
        translation=bridged_translation,
        rotation=bridged_rotation,
    )
        
    return stereo_A_C


class CameraArrayInitializer:
    def __init__(self, config_path: Path):

        logger.info("Creating initial estimate of camera array based on stereopairs...")

        self.config = rtoml.load(config_path)
        self.ports = self._get_ports()
        self.estimated_stereopairs = self._get_captured_stereopairs()
        self._fill_stereopair_gaps()
        # self.best_camera_array = self.get_best_camera_array()

    def _fill_stereopair_gaps(self):
        """
        Loop across missing pairs and create bridged stereopairs when possible.
        It may be that one iteration is not sufficient to fill all missing pairs,
        so iterate until no more missing pairs...
        
        The code below uses a naming convention to describe the relationship between
        two stereo pairs (A,X) and (X,C) that can be used to build a bridge stereopair (A,C)
        """

        # fill with dummy value to get the loop running
        missing_count_last_cycle = -1
        
        while len(self._get_missing_stereopairs()) != missing_count_last_cycle:
            
            # prep the variable. if it doesn't go down, terminate
            missing_count_last_cycle = len(self._get_missing_stereopairs())

            for pair in self._get_missing_stereopairs():
             
                port_A = pair[0]
                port_C = pair[1]
    
                # get lists of all the estimiated stereopairs that might bridge across test_missing
                all_pairs_A_X = [pair for pair in self.estimated_stereopairs.keys() if pair[0]==port_A]
                all_pairs_X_C = [pair for pair in self.estimated_stereopairs.keys() if pair[1]==port_C]
   
                stereopair_A_C = None

                for pair_A_X in all_pairs_A_X:
                    for pair_X_C in all_pairs_X_C:
                        if pair_A_X[1] == pair_X_C[0]:
                            # A bridge can be formed!
                            stereopair_A_X = self.estimated_stereopairs[pair_A_X]
                            stereopair_X_C = self.estimated_stereopairs[pair_X_C]
                            possible_stereopair_A_C = get_bridged_stereopair(stereopair_A_X, stereopair_X_C)
                            if stereopair_A_C is None:
                                # current possibility is better than nothing
                                stereopair_A_C = possible_stereopair_A_C
                            else:
                                # check if it's better than what you have already
                                # if it is, then overwrite the old one
                                if stereopair_A_C.error_score > possible_stereopair_A_C.error_score:
                                    stereopair_A_C = possible_stereopair_A_C

                if stereopair_A_C is not None:
                    self.add_stereopair(stereopair_A_C)

        if len(self._get_missing_stereopairs()) > 0:
            raise ValueError("Insufficient stereopairs to allow array to be estimated")

    def _get_missing_stereopairs(self):

        possible_stereopairs = [pair for pair in permutations(self.ports,2)]
        missing_stereopairs = [pair for pair in possible_stereopairs if pair not in self.estimated_stereopairs.keys()]

        return missing_stereopairs
        
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

    def _get_captured_stereopairs(self) -> dict:

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
            # NOTE: commenting out second conditional check below. If you come back to this in a month and 
            # things haven't been breaking, then just delete all these comments.
            if key.startswith("cam_"): # and not self.config[key]["ignore"]:
                port = data["port"]
                size = data["size"]
                rotation_count = data["rotation_count"]
                error = data["error"]
                matrix = np.array(data["matrix"], dtype=np.float64)
                distortions = np.array(data["distortions"], dtype=np.float64)
                # exposure = data["exposure"]
                grid_count = data["grid_count"]
                # ignore = data["ignore"]
                # verified_resolutions = data["verified_resolutions"]

                # update with extrinsics, though place anchor camera at origin
                if port == anchor_port:
                    translation = np.array([0, 0, 0], dtype=np.float64).T
                    rotation = np.array(
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
                    )
                else:
                    anchored_stereopair = self.estimated_stereopairs[(anchor_port, port)]
                    translation = anchored_stereopair.translation[:, 0]
                    rotation = anchored_stereopair.rotation
                    total_error_score += anchored_stereopair.error_score

                cam_data = CameraData(
                    port=port,
                    size=size,
                    rotation_count=rotation_count,
                    error=error,
                    matrix=matrix,
                    distortions=distortions,
                    # exposure,
                    grid_count=grid_count,
                    # ignore=ignore,
                    # verified_resolutions,
                    translation=translation,
                    rotation=rotation,
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
        for port in self.ports:
            array_error_score, camera_array = self._get_scored_anchored_array(port)
            array_error_scores[port] = array_error_score
            camera_arrays[port] = camera_array

        best_anchor = min(array_error_scores, key=array_error_scores.get)

        best_initial_array = camera_arrays[best_anchor]

        return best_initial_array

    def add_stereopair(self, stereopair:StereoPair):
        self.estimated_stereopairs[stereopair.pair] = stereopair
        inverted_stereopair = get_inverted_stereopair(stereopair)
        self.estimated_stereopairs[inverted_stereopair.pair] = inverted_stereopair
        

# def get_anchored_pairs(anchor: int, all_stereopairs:dict)->dict:


if __name__ == "__main__":

    session_directory = Path(__root__, "tests", "sessions", "217")

    config_path = Path(session_directory, "config.toml")

    initializer = CameraArrayInitializer(config_path)
        
    
    camera_array = initializer.get_best_camera_array()

    extrinsic_calibration_xy = Path(session_directory, "point_data.csv")

    point_estimates: PointEstimates = get_point_estimates(camera_array, extrinsic_calibration_xy)

    capture_volume = CaptureVolume(camera_array, point_estimates)

    pair_A_B = initializer.estimated_stereopairs[(0, 1)]
    pair_B_C = initializer.estimated_stereopairs[(1, 2)]

    bridged_pair = get_bridged_stereopair(pair_A_B, pair_B_C)
    logger.info(bridged_pair)

    # capture_volume.optimize()



# %%
