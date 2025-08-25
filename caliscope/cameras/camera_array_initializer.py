# %%


from collections import deque
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rtoml

import caliscope.logger
from caliscope import __root__
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.cameras.camera_array import CameraArray, CameraData

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


def get_bridged_stereopair(pair_A_B: StereoPair, pair_B_C: StereoPair) -> StereoPair:
    port_A = pair_A_B.primary_port
    port_C = pair_B_C.secondary_port

    A_B_error = pair_A_B.error_score
    B_C_error = pair_B_C.error_score
    A_C_error = A_B_error + B_C_error

    # new transformations are added on the left
    # https://youtube.com/watch?v=q0mRtuiKSKg&feature=shares&t=66
    bridged_transformation = np.matmul(pair_B_C.transformation, pair_A_B.transformation)
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
        logger.info("Creating initial estimate of camera array based on stereopairs contained in config.toml...")

        self.config = rtoml.load(config_path)
        self.ports = self._get_ports()
        self.estimated_stereopairs = self._get_captured_stereopairs()
        self._fill_stereopair_gaps()

    def _get_ports(self) -> list:
        """
        Gets all camera ports defined in the config file from the [cam_...] sections.
        This ensures that every camera is accounted for, even if it has no
        stereo relationships.
        """
        ports = []
        for key, params in self.config.items():
            if key.startswith("cam_"):
                ports.append(params["port"])

        ports.sort()  # ensure stable order
        return ports

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
            missing_stereo_pairs = self._get_missing_stereopairs()
            missing_count_last_cycle = len(missing_stereo_pairs)

            for pair in missing_stereo_pairs:
                port_A = pair[0]
                port_C = pair[1]

                # get lists of all the estimiated stereopairs that might bridge across test_missing
                all_pairs_A_X = [pair for pair in self.estimated_stereopairs.keys() if pair[0] == port_A]
                all_pairs_X_C = [pair for pair in self.estimated_stereopairs.keys() if pair[1] == port_C]

                best_bridge_pair = None

                for pair_A_X in all_pairs_A_X:
                    for pair_X_C in all_pairs_X_C:
                        if pair_A_X[1] == pair_X_C[0]:
                            # A bridge can be formed!
                            stereopair_A_X = self.estimated_stereopairs[pair_A_X]
                            stereopair_X_C = self.estimated_stereopairs[pair_X_C]
                            possible_stereopair_A_C = get_bridged_stereopair(stereopair_A_X, stereopair_X_C)
                            if (
                                best_bridge_pair is None
                                or best_bridge_pair.error_score > possible_stereopair_A_C.error_score
                            ):
                                best_bridge_pair = possible_stereopair_A_C

                if best_bridge_pair is not None:
                    self.add_stereopair(best_bridge_pair)

        if len(self._get_missing_stereopairs()) > 0:
            logger.warning("Could not form a fully connected camera graph. Some cameras will be unposed.")

    def _get_missing_stereopairs(self):
        possible_stereopairs = [pair for pair in permutations(self.ports, 2)]
        missing_stereopairs = [pair for pair in possible_stereopairs if pair not in self.estimated_stereopairs.keys()]

        return missing_stereopairs

    def _get_captured_stereopairs(self) -> Dict[Tuple[int, int], StereoPair]:
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

    def _find_largest_connected_component(self) -> set:
        """
        Performs a graph traversal to find all connected components (islands)
        of cameras and returns the one with the most members.
        """
        if not self.estimated_stereopairs:
            return set()

        # Build an adjacency list for the graph
        adj = {port: [] for port in self.ports}
        for port1, port2 in self.estimated_stereopairs.keys():
            adj[port1].append(port2)
            adj[port2].append(port1)

        visited = set()
        largest_component = set()

        for port in self.ports:
            if port not in visited:
                # Start a new traversal (BFS) for a new component
                current_component = set()
                q = deque([port])
                visited.add(port)

                while q:
                    u = q.popleft()
                    current_component.add(u)
                    for v in adj.get(u, []):
                        if v not in visited:
                            visited.add(v)
                            q.append(v)

                if len(current_component) > len(largest_component):
                    largest_component = current_component

        return largest_component

    def _get_scored_anchored_array(self, anchor_port: int) -> Tuple[float, CameraArray]:
        """
        Constructs a CameraArray anchored to a specific port.

        If a camera cannot be linked to the anchor, it is still included in the
        array but its `rotation` and `translation` are set to None.
        """
        cameras = {}
        total_error_score = 0

        for port in self.ports:
            data = self.config[f"cam_{port}"]

            translation = None
            rotation = None

            # update with extrinsics, though place anchor camera at origin
            if port == anchor_port:
                translation = np.array([0, 0, 0], dtype=np.float64).T
                rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
            else:
                pair_key = (anchor_port, port)
                if pair_key in self.estimated_stereopairs:
                    anchored_stereopair = self.estimated_stereopairs[pair_key]
                    translation = anchored_stereopair.translation[:, 0]
                    rotation = anchored_stereopair.rotation
                    total_error_score += anchored_stereopair.error_score

            # Create the CameraData object regardless of whether it's posed
            cameras[port] = CameraData(
                port=port,
                size=data["size"],
                rotation_count=data["rotation_count"],
                error=data["error"],
                matrix=np.array(data["matrix"], dtype=np.float64),
                distortions=np.array(data["distortions"], dtype=np.float64),
                grid_count=data["grid_count"],
                translation=translation,  # Will be None for unposed cameras
                rotation=rotation,  # Will be None for unposed cameras
            )

        camera_array = CameraArray(cameras)

        return total_error_score, camera_array

    def get_best_camera_array(self):
        """
        Finds the largest connected group of cameras and returns the camera array
        anchored from the node within that group that has the lowest cumulative error.
        """
        # STEP 1: Find the largest connected component of cameras.
        # this helps to ensure the largest "island" of cameras is linked if pairs exist with not bridge
        main_group_ports = self._find_largest_connected_component()

        if not main_group_ports:
            logger.warning("No connected stereo pairs found. Returning an array with all cameras unposed.")
            # Use a dummy anchor_port; it won't be used to pose anything.
            _, unposed_array = self._get_scored_anchored_array(anchor_port=-1, all_ports=self.ports)
            return unposed_array

        logger.info(
            f"Identified main camera group with {len(main_group_ports)} members: {sorted(list(main_group_ports))}"
        )

        # STEP 2: Find the best anchor *within the main group*.
        array_error_scores = {}
        camera_arrays = {}

        for port in main_group_ports:
            array_error_score, camera_array = self._get_scored_anchored_array(port)
            # The score is only non-zero if it can connect to others, which is guaranteed for this group.
            array_error_scores[port] = array_error_score
            camera_arrays[port] = camera_array

        best_anchor = min(array_error_scores, key=array_error_scores.get)
        best_initial_array = camera_arrays[best_anchor]

        unposed_count = len(best_initial_array.unposed_cameras)
        if unposed_count > 0:
            unposed_ports = list(best_initial_array.unposed_cameras.keys())
            logger.warning(f"{unposed_count} cameras are not in the main group and remain unposed: {unposed_ports}")

        logger.info(f"Selected camera {best_anchor} as anchor, yielding lowest initial error.")

        return best_initial_array

    def add_stereopair(self, stereopair: StereoPair):
        self.estimated_stereopairs[stereopair.pair] = stereopair
        inverted_stereopair = get_inverted_stereopair(stereopair)
        self.estimated_stereopairs[inverted_stereopair.pair] = inverted_stereopair


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
