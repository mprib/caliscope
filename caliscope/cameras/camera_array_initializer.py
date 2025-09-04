# caliscope/cameras/camera_array_initializer.py

from collections import deque
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Tuple

import numpy as np

import caliscope.logger
from caliscope.cameras.camera_array import CameraArray, CameraData

logger = caliscope.logger.get(__name__)


@dataclass
class StereoPair:
    """
    A dataclass to hold the extrinsic parameters associated with the cv2.stereoCalibrate
    function output. Additionally provides some convenience methods to get common transformations
    of the data.
    """

    primary_port: int
    secondary_port: int
    error_score: float
    translation: np.ndarray
    rotation: np.ndarray

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.primary_port, self.secondary_port)

    @property
    def transformation(self) -> np.ndarray:
        R_stack = np.vstack([self.rotation, np.array([0, 0, 0])])
        t_stack = np.vstack([self.translation.reshape(3, 1), np.array([[1]])])
        return np.hstack([R_stack, t_stack])


def get_inverted_stereopair(stereo_pair: StereoPair) -> StereoPair:
    inverted_transformation = np.linalg.inv(stereo_pair.transformation)
    return StereoPair(
        primary_port=stereo_pair.secondary_port,
        secondary_port=stereo_pair.primary_port,
        error_score=stereo_pair.error_score,
        rotation=inverted_transformation[0:3, 0:3],
        translation=inverted_transformation[0:3, 3:],
    )


def get_bridged_stereopair(pair_A_B: StereoPair, pair_B_C: StereoPair) -> StereoPair:
    port_A = pair_A_B.primary_port
    port_C = pair_B_C.secondary_port
    A_C_error = pair_A_B.error_score + pair_B_C.error_score

    bridged_transformation = np.matmul(pair_B_C.transformation, pair_A_B.transformation)
    return StereoPair(
        primary_port=port_A,
        secondary_port=port_C,
        error_score=A_C_error,
        rotation=bridged_transformation[0:3, 0:3],
        translation=bridged_transformation[0:3, 3:],
    )


class CameraArrayInitializer:
    """
    A pure solver that determines a globally consistent set of camera poses
    from a collection of pairwise stereo calibration results.
    """

    # REFACTOR: Constructor now accepts data directly, not a config path.
    def __init__(self, camera_array: CameraArray, stereo_results: Dict[str, Dict]):
        logger.info("Creating initial estimate of camera array based on stereo results...")
        self.camera_array = camera_array
        self.ports = sorted(list(self.camera_array.cameras.keys()))
        self.estimated_stereopairs = self._parse_stereo_results(stereo_results)
        self._fill_stereopair_gaps()

    # REFACTOR: New method to parse the raw dictionary from StereoCalibrator
    def _parse_stereo_results(self, stereo_results: Dict[str, Dict]) -> Dict[Tuple[int, int], StereoPair]:
        """Parses the raw stereo results dictionary into StereoPair objects."""
        stereopairs = {}
        for key, params in stereo_results.items():
            # key is 'stereo_1_2'
            _, port_A_str, port_B_str = key.split("_")
            port_A, port_B = int(port_A_str), int(port_B_str)

            new_stereopair = StereoPair(
                primary_port=port_A,
                secondary_port=port_B,
                error_score=float(params["RMSE"]),
                rotation=np.array(params["rotation"], dtype=np.float64),
                translation=np.array(params["translation"], dtype=np.float64),
            )
            stereopairs[new_stereopair.pair] = new_stereopair

        inverted_stereopairs = {
            get_inverted_stereopair(sp).pair: get_inverted_stereopair(sp) for sp in stereopairs.values()
        }
        return {**stereopairs, **inverted_stereopairs}

    def _fill_stereopair_gaps(self):
        """Iteratively bridge gaps in the stereo pair graph."""
        missing_count_last_cycle = -1
        while len(self._get_missing_stereopairs()) != missing_count_last_cycle:
            missing_stereo_pairs = self._get_missing_stereopairs()
            missing_count_last_cycle = len(missing_stereo_pairs)

            for port_A, port_C in missing_stereo_pairs:
                best_bridge_pair = None
                for port_X in self.ports:
                    pair_A_X_key = (port_A, port_X)
                    pair_X_C_key = (port_X, port_C)

                    if pair_A_X_key in self.estimated_stereopairs and pair_X_C_key in self.estimated_stereopairs:
                        stereopair_A_X = self.estimated_stereopairs[pair_A_X_key]
                        stereopair_X_C = self.estimated_stereopairs[pair_X_C_key]
                        possible_bridge = get_bridged_stereopair(stereopair_A_X, stereopair_X_C)

                        if best_bridge_pair is None or best_bridge_pair.error_score > possible_bridge.error_score:
                            best_bridge_pair = possible_bridge

                if best_bridge_pair is not None:
                    self.add_stereopair(best_bridge_pair)

        if len(self._get_missing_stereopairs()) > 0:
            logger.warning("Could not form a fully connected camera graph. Some cameras will be unposed.")

    def _get_missing_stereopairs(self) -> list[Tuple[int, int]]:
        possible_stereopairs = list(permutations(self.ports, 2))
        return [pair for pair in possible_stereopairs if pair not in self.estimated_stereopairs]

    def _find_largest_connected_component(self) -> set:
        """Finds the largest connected subgraph of cameras."""
        if not self.estimated_stereopairs:
            return set()

        adj = {port: [] for port in self.ports}
        for port1, port2 in self.estimated_stereopairs.keys():
            adj[port1].append(port2)

        visited = set()
        largest_component = set()
        for port in self.ports:
            if port not in visited:
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

    # REFACTOR: Now builds a new CameraArray from the input one, adding extrinsics.
    def _get_scored_anchored_array(self, anchor_port: int) -> Tuple[float, CameraArray]:
        """Constructs a CameraArray anchored to a specific port."""
        posed_cameras = {}
        total_error_score = 0.0

        # Start with a deep copy of the original cameras dict
        for port in self.ports:
            original_cam_data = self.camera_array.cameras[port]

            # Use all original intrinsic properties
            posed_cameras[port] = CameraData(
                port=original_cam_data.port,
                size=original_cam_data.size,
                rotation_count=original_cam_data.rotation_count,
                error=original_cam_data.error,
                matrix=original_cam_data.matrix,
                distortions=original_cam_data.distortions,
                grid_count=original_cam_data.grid_count,
                # Extrinsics will be set below
                translation=None,
                rotation=None,
            )

        # Set the anchor camera's pose to the origin
        posed_cameras[anchor_port].rotation = np.eye(3, dtype=np.float64)
        posed_cameras[anchor_port].translation = np.zeros(3, dtype=np.float64)

        # Pose other cameras relative to the anchor
        for port in self.ports:
            if port == anchor_port:
                continue

            pair_key = (anchor_port, port)
            if pair_key in self.estimated_stereopairs:
                anchored_stereopair = self.estimated_stereopairs[pair_key]
                posed_cameras[port].translation = anchored_stereopair.translation.flatten()
                posed_cameras[port].rotation = anchored_stereopair.rotation
                total_error_score += anchored_stereopair.error_score

        return total_error_score, CameraArray(posed_cameras)

    def get_best_camera_array(self) -> CameraArray:
        """
        Finds the largest connected group of cameras and returns the camera array
        anchored from the node within that group that has the lowest cumulative error.
        """
        main_group_ports = self._find_largest_connected_component()

        if not main_group_ports:
            logger.warning("No connected stereo pairs found. Returning the original array with no cameras posed.")
            return self.camera_array

        logger.info(f"Identified main camera group: {sorted(list(main_group_ports))}")

        best_anchor = -1
        lowest_error = float("inf")
        best_initial_array = None

        for port in main_group_ports:
            array_error_score, camera_array = self._get_scored_anchored_array(port)
            if array_error_score < lowest_error:
                lowest_error = array_error_score
                best_anchor = port
                best_initial_array = camera_array

        if best_initial_array is None:
            # This case should be rare, but as a fallback return the original array
            logger.error("Could not determine a best anchor. Returning original unposed array.")
            return self.camera_array

        unposed_ports = list(best_initial_array.unposed_cameras.keys())
        if unposed_ports:
            logger.warning(f"Cameras not in the main group remain unposed: {unposed_ports}")

        logger.info(f"Selected camera {best_anchor} as anchor, yielding lowest initial error.")
        return best_initial_array

    def add_stereopair(self, stereopair: StereoPair):
        """Adds a new stereo pair and its inverse to the collection."""
        self.estimated_stereopairs[stereopair.pair] = stereopair
        inverted = get_inverted_stereopair(stereopair)
        self.estimated_stereopairs[inverted.pair] = inverted
