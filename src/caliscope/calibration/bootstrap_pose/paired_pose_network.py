from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple
from itertools import permutations

import numpy as np

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.calibration.bootstrap_pose.stereopairs import StereoPair

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairedPoseNetwork:
    """Immutable graph of stereo pair relationships between cameras."""

    _pairs: Dict[Tuple[int, int], StereoPair]

    @classmethod
    def from_raw_estimates(cls, raw_pairs: Dict[Tuple[int, int], StereoPair]) -> PairedPoseNetwork:
        """
        Create a StereoPairGraph from raw estimates and fill in missing pairs
        by bridging through intermediate cameras.
        Matches the legacy CameraArrayInitializer logic exactly.
        """
        # Initialize with raw pairs
        # Legacy behavior: inverted pairs are added to the dictionary
        all_pairs = raw_pairs.copy()
        inverted_pairs = {cls._invert_pair(pair).pair: cls._invert_pair(pair) for pair in all_pairs.values()}
        all_pairs.update(inverted_pairs)

        # Get all ports involved
        ports_set = set()
        for a, b in all_pairs.keys():
            ports_set.add(a)
            ports_set.add(b)

        # Sort ports to ensure deterministic graph construction
        # Legacy code used: sorted(list(self.camera_array.cameras.keys()))
        ports = sorted(list(ports_set))

        # Iteratively fill gaps using legacy permutation logic
        missing_count_last_cycle = -1

        while True:
            # Replicate _get_missing_stereopairs()
            possible_pairs = list(permutations(ports, 2))
            missing_pairs = [pair for pair in possible_pairs if pair not in all_pairs]

            current_missing = len(missing_pairs)

            if current_missing == missing_count_last_cycle:
                break

            if current_missing == 0:
                break

            missing_count_last_cycle = current_missing

            # Legacy logic: Iterate through missing pairs (A, C)
            for port_a, port_c in missing_pairs:
                best_bridge = None

                # Legacy logic: Iterate through potential bridges X in sorted order
                for port_x in ports:
                    pair_a_x_key = (port_a, port_x)
                    pair_x_c_key = (port_x, port_c)

                    if pair_a_x_key in all_pairs and pair_x_c_key in all_pairs:
                        pair_a_x = all_pairs[pair_a_x_key]
                        pair_x_c = all_pairs[pair_x_c_key]

                        possible_bridge = cls._bridge_pairs(pair_a_x, pair_x_c)

                        # Legacy Comparison: if best is None or old > new (strictly greater)
                        if best_bridge is None or best_bridge.error_score > possible_bridge.error_score:
                            best_bridge = possible_bridge

                if best_bridge is not None:
                    # Add both directions immediately, matching legacy add_stereopair()
                    all_pairs[best_bridge.pair] = best_bridge
                    inverted = cls._invert_pair(best_bridge)
                    all_pairs[inverted.pair] = inverted
        # Before: return cls(_pairs=all_pairs)
        # Add:
        logger.info(f"StereoPairGraph created with {len(all_pairs)} pairs")
        # Count bridged pairs (those with accumulated error > raw RMSE range)
        raw_errors = [p.error_score for p in raw_pairs.values()]
        max_raw = max(raw_errors) if raw_errors else 0
        bridged_count = sum(1 for p in all_pairs.values() if p.error_score > max_raw * 1.5)
        logger.info(f"  Estimated bridged pairs: {bridged_count}")

        return cls(_pairs=all_pairs)

    def _build_anchored_config(
        self, camera_array: CameraArray, anchor_port: int
    ) -> tuple[float, Dict[int, CameraData]]:
        """
        Builds a camera configuration anchored to the specified port.
        Uses direct lookup from the anchor node, relying on the gap-filling step
        to have created the necessary edges.
        """
        total_error_score = 0.0
        configured_cameras = {}

        # Get sorted ports to match legacy iteration order
        ports = sorted(list(camera_array.cameras.keys()))

        # Create new CameraData objects (legacy _get_scored_anchored_array behavior)
        for port, cam_data in camera_array.cameras.items():
            configured_cameras[port] = CameraData(
                port=cam_data.port,
                size=cam_data.size,
                rotation_count=cam_data.rotation_count,
                error=cam_data.error,
                matrix=cam_data.matrix,
                distortions=cam_data.distortions,
                grid_count=cam_data.grid_count,
                exposure=cam_data.exposure,
                ignore=cam_data.ignore,
                fisheye=cam_data.fisheye,
                translation=None,
                rotation=None,
            )

        # Set anchor to origin
        configured_cameras[anchor_port].rotation = np.eye(3, dtype=np.float64)
        configured_cameras[anchor_port].translation = np.zeros(3, dtype=np.float64)

        # Pose other cameras using direct lookup
        for port in ports:
            if port == anchor_port:
                continue

            # Legacy: Direct lookup Anchor -> Port
            pair_key = (anchor_port, port)

            if pair_key in self._pairs:
                anchored_stereopair = self._pairs[pair_key]

                # Apply transformation
                configured_cameras[port].translation = anchored_stereopair.translation.flatten()
                configured_cameras[port].rotation = anchored_stereopair.rotation

                # Accumulate error
                total_error_score += anchored_stereopair.error_score

        return total_error_score, configured_cameras

    @staticmethod
    def _bridge_pairs(pair_ab: StereoPair, pair_bc: StereoPair) -> StereoPair:
        """Create a bridged pair A->C from A->B and B->C."""
        # Transform composition: A->C = B->C * A->B
        trans_ab = pair_ab.transformation
        trans_bc = pair_bc.transformation
        trans_ac = np.matmul(trans_bc, trans_ab)

        return StereoPair(
            primary_port=pair_ab.primary_port,
            secondary_port=pair_bc.secondary_port,
            error_score=pair_ab.error_score + pair_bc.error_score,
            rotation=trans_ac[0:3, 0:3],
            translation=trans_ac[0:3, 3:],
        )

    def get_pair(self, port_a: int, port_b: int) -> StereoPair | None:
        """Retrieve a stereo pair by port pair, returns None if not found."""
        return self._pairs.get((port_a, port_b))

    def get_best_anchored_camera_array(
        self, main_group_ports, camera_array
    ) -> tuple[int, Dict[int, CameraData]] | tuple[None, Dict[int, CameraData]]:
        # Find best anchor by trying each port in the main group
        best_anchor = -1
        lowest_error = float("inf")
        best_cameras_config = None

        logger.info("Assessing best port to anchor camera array")
        for port in main_group_ports:
            error_score, cameras_config = self._build_anchored_config(camera_array, port)
            logger.info(f"    port {port} anchor_score = {error_score}")
            if error_score < lowest_error:
                lowest_error = error_score
                best_anchor = port
                best_cameras_config = cameras_config

        if best_anchor == -1:
            return None, camera_array.cameras
        else:
            return best_anchor, best_cameras_config

    def apply_to(self, camera_array: CameraArray, anchor_cam: int | None = None) -> None:
        """
        Mutates camera_array in place by solving for globally consistent camera poses
        from the stereo pair graph.
        """

        ports = sorted(camera_array.cameras.keys())
        # Find largest connected component (Legacy behavior used this to filter main group)
        main_group_ports = self._find_largest_connected_component(ports)

        if anchor_cam:
            error_score, best_cameras_config = self._build_anchored_config(camera_array, anchor_cam)
        else:
            anchor_cam, best_cameras_config = self.get_best_anchored_camera_array(main_group_ports, camera_array)
            logger.info(f"Selected camera {anchor_cam} as anchor, yielding lowest initial error.")

        logger.info("Applying stereo pair graph to camera array...")

        # Apply the best configuration to the original camera array
        for port, cam_data in best_cameras_config.items():
            camera_array.cameras[port] = cam_data

        unposed_ports = [p for p in ports if p not in main_group_ports]
        if unposed_ports:
            logger.warning(f"Cameras not in the main group remain unposed: {unposed_ports}")

    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Dict]) -> PairedPoseNetwork:
        """Alternative constructor for backward compatibility with legacy dictionary format."""
        pairs = {}
        for key, params in data.items():
            # key is 'stereo_1_2'
            _, port_a_str, port_b_str = key.split("_")
            port_a, port_b = int(port_a_str), int(port_b_str)

            pair = StereoPair(
                primary_port=port_a,
                secondary_port=port_b,
                error_score=float(params["RMSE"]),
                rotation=np.array(params["rotation"], dtype=np.float64),
                translation=np.array(params["translation"], dtype=np.float64),
            )
            pairs[pair.pair] = pair

        return cls.from_raw_estimates(pairs)

    def to_dict(self) -> Dict[str, Dict]:
        """Serialize to legacy dictionary format."""
        return {
            f"stereo_{a}_{b}": {
                "rotation": pair.rotation.tolist(),
                "translation": pair.translation.tolist(),
                "RMSE": pair.error_score,
            }
            for (a, b), pair in self._pairs.items()
            if a < b  # Only store forward pairs to avoid duplication
        }

    @staticmethod
    def _invert_pair(stereo_pair: StereoPair) -> StereoPair:
        """Helper to create inverted stereo pair."""
        inverted_transformation = np.linalg.inv(stereo_pair.transformation)
        return StereoPair(
            primary_port=stereo_pair.secondary_port,
            secondary_port=stereo_pair.primary_port,
            error_score=stereo_pair.error_score,
            rotation=inverted_transformation[0:3, 0:3],
            translation=inverted_transformation[0:3, 3:],
        )

    def _find_largest_connected_component(self, ports: list[int]) -> set[int]:
        """Finds the largest connected subgraph of cameras."""
        if not self._pairs:
            return set()

        adj = {port: [] for port in ports}
        for port1, port2 in self._pairs.keys():
            if port1 in adj:
                adj[port1].append(port2)

        visited = set()
        largest_component = set()
        for port in ports:
            if port not in visited:
                current_component = set()
                # deque([port]) used in legacy
                from collections import deque

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
