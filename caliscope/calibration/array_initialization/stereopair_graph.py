"""
caliscope/calibration/array_initialization/stereopair_graph.py
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.calibration.array_initialization.stereopairs import StereoPair

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StereoPairGraph:
    """Immutable graph of stereo pair relationships between cameras."""

    _pairs: Dict[Tuple[int, int], StereoPair]

    def __post_init__(self):
        """Validate and fill gaps in the graph after initialization."""
        # This is called automatically after __init__ for frozen dataclasses
        # We can't modify self._pairs directly, so we need to do gap-filling
        # in a classmethod constructor instead
        pass

    @classmethod
    def from_raw_estimates(cls, raw_pairs: Dict[Tuple[int, int], StereoPair]) -> StereoPairGraph:
        """
        Create a StereoPairGraph from raw estimates and fill in missing pairs
        by bridging through intermediate cameras.
        """
        logger.info(f"{'=' * 60}")
        logger.info("from_raw_estimates: START")
        logger.info(f"{'=' * 60}")
        logger.info(f"Initial raw pairs: {len(raw_pairs)}")

        all_pairs = raw_pairs.copy()

        # Get all ports involved
        ports_set = set()
        for a, b in all_pairs.keys():
            ports_set.add(a)
            ports_set.add(b)

        # Sort ports to ensure deterministic graph construction
        # This matches the behavior of the legacy CameraArrayInitializer
        ports = sorted(list(ports_set))

        logger.info(f"Discovered ports: {ports}")

        # Add inverted pairs first (like old CameraArrayInitializer did)
        inverted_pairs = {cls._invert_pair(pair).pair: cls._invert_pair(pair) for pair in all_pairs.values()}
        all_pairs.update(inverted_pairs)
        logger.info(f"After adding inverses: {len(all_pairs)} pairs")

        # Iteratively fill gaps (using proven logic from old implementation)
        missing_count_last_cycle = -1
        iteration = 0
        max_iterations = len(ports) * len(ports)  # Safety limit

        while iteration < max_iterations:
            missing_pairs = cls._get_missing_pairs(all_pairs, ports)
            current_missing = len(missing_pairs)

            logger.info(f"Iteration {iteration}: {current_missing} missing pairs")

            if current_missing == 0:
                logger.info("✓ All pairs filled successfully!")
                break

            if current_missing == missing_count_last_cycle:
                logger.warning(f"✗ No progress made, stopping early. Missing: {missing_pairs}")
                break

            missing_count_last_cycle = current_missing
            bridges_created = 0

            # Try to bridge each missing pair
            for port_a, port_c in missing_pairs:
                best_bridge = None
                best_bridge_b = None

                # Try all possible intermediate ports
                for port_b in ports:
                    pair_a_b = (port_a, port_b)
                    pair_b_c = (port_b, port_c)

                    if pair_a_b in all_pairs and pair_b_c in all_pairs:
                        # Can bridge through port_b
                        bridged = cls._bridge_pairs(all_pairs[pair_a_b], all_pairs[pair_b_c])

                        if best_bridge is None or bridged.error_score < best_bridge.error_score:
                            best_bridge = bridged
                            best_bridge_b = port_b

                if best_bridge is not None:
                    # Add both directions
                    all_pairs[best_bridge.pair] = best_bridge
                    inverted = cls._invert_pair(best_bridge)
                    all_pairs[inverted.pair] = inverted
                    bridges_created += 1
                    logger.debug(
                        f"  Bridged {port_a}->{port_c} via {best_bridge_b}: error={best_bridge.error_score:.4f}"
                    )

            logger.info(f"  → Created {bridges_created} bridges this iteration")
            iteration += 1

        if iteration >= max_iterations:
            logger.error("✗ Hit maximum iterations, gap-filling incomplete")

        logger.info(f"{'=' * 60}")
        logger.info(f"Final graph: {len(all_pairs)} pairs")
        logger.info(f"Sample pairs: {list(all_pairs.keys())[:10]}")
        logger.info(f"{'=' * 60}")

        return cls(_pairs=all_pairs)

    def _build_anchored_config(
        self, camera_array: CameraArray, anchor_port: int
    ) -> tuple[float, Dict[int, CameraData]]:
        """Builds a camera configuration anchored to the specified port."""
        # logger.info(f"{'=' * 60}")
        # logger.info(f"Building anchored config for anchor={anchor_port}")
        # logger.info(f"{'=' * 60}")

        total_error_score = 0.0
        configured_cameras = {}

        # Create new CameraData objects
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
        logger.info(f"✓ Anchor camera {anchor_port} set to origin")

        # Pose other cameras using BFS traversal
        visited = {anchor_port}
        queue = deque([anchor_port])
        traversal_steps = 0

        while queue:
            current_port = queue.popleft()
            current_cam = configured_cameras[current_port]

            logger.debug(f"Traversing from camera {current_port}")

            # Find all neighbors of current camera
            for (src, dst), stereo_pair in self._pairs.items():
                if src == current_port and dst not in visited:
                    # Compose transformation: anchor->dst = anchor->src @ src->dst
                    if current_cam.rotation is not None and current_cam.translation is not None:
                        # Build 4x4 transformation matrix for current camera
                        T_anchor_src = np.eye(4)
                        T_anchor_src[0:3, 0:3] = current_cam.rotation
                        T_anchor_src[0:3, 3] = current_cam.translation

                        # Get src->dst transformation from stereo pair
                        T_src_dst = stereo_pair.transformation

                        # Compose: T_anchor_dst = T_anchor_src @ T_src_dst
                        T_anchor_dst = np.matmul(T_anchor_src, T_src_dst)

                        # Update neighbor's pose
                        configured_cameras[dst].rotation = T_anchor_dst[0:3, 0:3]
                        configured_cameras[dst].translation = T_anchor_dst[0:3, 3]

                        # Accumulate error
                        total_error_score += stereo_pair.error_score

                        # Mark as visited and add to queue
                        visited.add(dst)
                        queue.append(dst)
                        traversal_steps += 1

                        logger.debug(
                            f"  ✓ Posed camera {dst} via {src}: "
                            f"error={stereo_pair.error_score:.4f}, "
                            f"visited={len(visited)}/{len(configured_cameras)}"
                        )

        logger.info(f"{'=' * 60}")
        logger.info(f"Traversal complete: {traversal_steps} steps, {len(visited)} cameras posed")
        logger.info(f"Total error score: {total_error_score:.4f}")

        # Log unposed cameras
        unposed = [p for p in configured_cameras.keys() if configured_cameras[p].rotation is None]
        if unposed:
            logger.warning(f"Unposed cameras: {unposed}")
        else:
            logger.info("✓ All cameras successfully posed")
        logger.info(f"{'=' * 60}")

        return total_error_score, configured_cameras

    @staticmethod
    def _get_missing_pairs(pairs: Dict[Tuple[int, int], StereoPair], ports: List[int]) -> list[Tuple[int, int]]:
        """Find all missing directed pairs in the graph."""
        missing = []
        for a in ports:
            for b in ports:
                if a != b and (a, b) not in pairs:
                    missing.append((a, b))
        return missing

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

    def apply_to(self, camera_array: CameraArray) -> None:
        """
        Mutates camera_array in place by solving for globally consistent camera poses
        from the stereo pair graph.
        """
        logger.info("Applying stereo pair graph to camera array...")

        ports = sorted(camera_array.cameras.keys())

        # Find largest connected component
        main_group_ports = self._find_largest_connected_component(ports)

        if not main_group_ports:
            logger.warning("No connected stereo pairs found. No cameras will be posed.")
            return

        logger.info(f"Identified main camera group: {sorted(list(main_group_ports))}")

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

        if best_cameras_config is None:
            logger.error("Could not determine a best anchor. No cameras will be posed.")
            return

        # Apply the best configuration to the original camera array
        for port, cam_data in best_cameras_config.items():
            camera_array.cameras[port] = cam_data

        unposed_ports = [p for p in ports if p not in main_group_ports]
        if unposed_ports:
            logger.warning(f"Cameras not in the main group remain unposed: {unposed_ports}")

        logger.info(f"Selected camera {best_anchor} as anchor, yielding lowest initial error.")

    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Dict]) -> StereoPairGraph:
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

        # We delegate to from_raw_estimates to ensure full graph construction (inverses + bridges)
        # matching the behavior of the legacy CameraArrayInitializer
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
