from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
from itertools import permutations

import cv2
import numpy as np
import rtoml

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.bootstrap_pose.stereopairs import StereoPair
from caliscope.core.toml_helpers import _list_to_array

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
        inverted_pairs = {(inv := pair.inverted()).pair: inv for pair in all_pairs.values()}
        all_pairs.update(inverted_pairs)

        # Get all cam_ids involved
        cam_ids_set = set()
        for a, b in all_pairs.keys():
            cam_ids_set.add(a)
            cam_ids_set.add(b)

        # Sort cam_ids to ensure deterministic graph construction
        # Legacy code used: sorted(list(self.camera_array.cameras.keys()))
        cam_ids = sorted(list(cam_ids_set))

        # Iteratively fill gaps using legacy permutation logic
        missing_count_last_cycle = -1

        while True:
            # Replicate _get_missing_stereopairs()
            possible_pairs = list(permutations(cam_ids, 2))
            missing_pairs = [pair for pair in possible_pairs if pair not in all_pairs]

            current_missing = len(missing_pairs)

            if current_missing == missing_count_last_cycle:
                break

            if current_missing == 0:
                break

            missing_count_last_cycle = current_missing

            # Legacy logic: Iterate through missing pairs (A, C)
            for cam_id_a, cam_id_c in missing_pairs:
                best_bridge = None

                # Legacy logic: Iterate through potential bridges X in sorted order
                for cam_id_x in cam_ids:
                    pair_a_x_key = (cam_id_a, cam_id_x)
                    pair_x_c_key = (cam_id_x, cam_id_c)

                    if pair_a_x_key in all_pairs and pair_x_c_key in all_pairs:
                        pair_a_x = all_pairs[pair_a_x_key]
                        pair_x_c = all_pairs[pair_x_c_key]

                        possible_bridge = pair_a_x.link(pair_x_c)

                        # Legacy Comparison: if best is None or old > new (strictly greater)
                        if best_bridge is None or best_bridge.error_score > possible_bridge.error_score:
                            best_bridge = possible_bridge

                if best_bridge is not None:
                    # Add both directions immediately, matching legacy add_stereopair()
                    all_pairs[best_bridge.pair] = best_bridge
                    inverted = best_bridge.inverted()
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
        self, camera_array: CameraArray, anchor_cam_id: int
    ) -> tuple[float, Dict[int, CameraData]]:
        """
        Builds a camera configuration anchored to the specified cam_id.
        Uses direct lookup from the anchor node, relying on the gap-filling step
        to have created the necessary edges.
        """
        total_error_score = 0.0
        configured_cameras = {}

        # Get sorted cam_ids to match legacy iteration order
        cam_ids = sorted(list(camera_array.cameras.keys()))

        # Create new CameraData objects (legacy _get_scored_anchored_array behavior)
        for cam_id, cam_data in camera_array.cameras.items():
            configured_cameras[cam_id] = CameraData(
                cam_id=cam_data.cam_id,
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
        configured_cameras[anchor_cam_id].rotation = np.eye(3, dtype=np.float64)
        configured_cameras[anchor_cam_id].translation = np.zeros(3, dtype=np.float64)

        # Pose other cameras using direct lookup
        for cam_id in cam_ids:
            if cam_id == anchor_cam_id:
                continue

            # Legacy: Direct lookup Anchor -> cam_id
            pair_key = (anchor_cam_id, cam_id)

            if pair_key in self._pairs:
                anchored_stereopair = self._pairs[pair_key]

                # Apply transformation
                configured_cameras[cam_id].translation = anchored_stereopair.translation.flatten()
                configured_cameras[cam_id].rotation = anchored_stereopair.rotation

                # Accumulate error
                total_error_score += anchored_stereopair.error_score

        return total_error_score, configured_cameras

    def get_pair(self, cam_id_a: int, cam_id_b: int) -> StereoPair | None:
        """Retrieve a stereo pair by cam_id pair, returns None if not found."""
        return self._pairs.get((cam_id_a, cam_id_b))

    def get_best_anchored_camera_array(
        self, main_group_cam_ids, camera_array
    ) -> tuple[int, Dict[int, CameraData]] | tuple[None, Dict[int, CameraData]]:
        # Find best anchor by trying each cam_id in the main group
        best_anchor = -1
        lowest_error = float("inf")
        best_cameras_config = None

        logger.info("Assessing best cam_id to anchor camera array")
        for cam_id in main_group_cam_ids:
            error_score, cameras_config = self._build_anchored_config(camera_array, cam_id)
            logger.info(f"    cam_id {cam_id} anchor_score = {error_score}")
            if error_score < lowest_error:
                lowest_error = error_score
                best_anchor = cam_id
                best_cameras_config = cameras_config

        if best_anchor == -1:
            return None, camera_array.cameras
        else:
            # best_cameras_config is guaranteed to be set if best_anchor != -1
            assert best_cameras_config is not None
            return best_anchor, best_cameras_config

    def apply_to(self, camera_array: CameraArray, anchor_cam: int | None = None) -> None:
        """
        Mutates camera_array in place by solving for globally consistent camera poses
        from the stereo pair graph.
        """

        cam_ids = sorted(camera_array.cameras.keys())
        # Find largest connected component (Legacy behavior used this to filter main group)
        main_group_cam_ids = self._find_largest_connected_component(cam_ids)

        if anchor_cam:
            error_score, best_cameras_config = self._build_anchored_config(camera_array, anchor_cam)
        else:
            anchor_cam, best_cameras_config = self.get_best_anchored_camera_array(main_group_cam_ids, camera_array)
            logger.info(f"Selected camera {anchor_cam} as anchor, yielding lowest initial error.")

        logger.info("Applying stereo pair graph to camera array...")

        # Apply the best configuration to the original camera array
        for cam_id, cam_data in best_cameras_config.items():
            camera_array.cameras[cam_id] = cam_data

        unposed_cam_ids = [c for c in cam_ids if c not in main_group_cam_ids]
        if unposed_cam_ids:
            logger.warning(f"Cameras not in the main group remain unposed: {unposed_cam_ids}")

    @classmethod
    def from_toml(cls, path: Path) -> PairedPoseNetwork:
        """Load PairedPoseNetwork from TOML file.

        The file stores only directly calibrated stereo pairs (primary_cam_id <
        secondary_cam_id) with Rodrigues rotation vectors. On load:
        1. Converts Rodrigues vectors back to 3x3 rotation matrices
        2. Reconstructs full graph via from_raw_estimates()

        Raises:
            PersistenceError: If file doesn't exist or format is invalid
        """
        from caliscope.persistence import PersistenceError

        if not path.exists():
            raise PersistenceError(f"Stereo pairs file not found: {path}")

        try:
            data = rtoml.load(path)
            if not data:
                return cls({})

            raw_pairs = {}
            for key, pair_data in data.items():
                try:
                    _, cam_id_a_str, cam_id_b_str = key.split("_")
                    cam_id_a, cam_id_b = int(cam_id_a_str), int(cam_id_b_str)
                except (ValueError, AttributeError):
                    logger.warning(f"Skipping invalid stereo pair key: {key}")
                    continue

                rotation_rodrigues = _list_to_array(pair_data.get("rotation"))
                if rotation_rodrigues is None:
                    logger.warning(f"Missing rotation for pair {key}, skipping")
                    continue
                if rotation_rodrigues.shape != (3,):
                    logger.warning(f"Invalid rotation shape for pair {key}: {rotation_rodrigues.shape}, expected (3,)")
                    continue

                rotation_matrix = cv2.Rodrigues(rotation_rodrigues)[0]

                translation = _list_to_array(pair_data.get("translation"))
                if translation is None:
                    logger.warning(f"Missing translation for pair {key}, skipping")
                    continue
                if translation.shape not in [(3, 1), (3,)]:
                    logger.warning(
                        f"Invalid translation shape for pair {key}: {translation.shape}, expected (3,1) or (3,)"
                    )
                    continue

                if translation.shape == (3,):
                    translation = translation.reshape(3, 1)

                pair = StereoPair(
                    primary_cam_id=cam_id_a,
                    secondary_cam_id=cam_id_b,
                    error_score=float(pair_data.get("RMSE", 0.0)),
                    rotation=rotation_matrix,
                    translation=translation,
                )
                raw_pairs[pair.pair] = pair

            return cls.from_raw_estimates(raw_pairs)

        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(f"Failed to load stereo pairs from {path}: {e}") from e

    def to_toml(self, path: Path) -> None:
        """Save PairedPoseNetwork to TOML file.

        Only stores forward pairs (primary < secondary) to avoid duplication.
        Converts 3x3 rotation matrices to 3x1 Rodrigues vectors.

        Raises:
            PersistenceError: If write fails
        """
        from caliscope.persistence import PersistenceError, _safe_write_toml

        try:
            stereo_data = {}
            for (a, b), pair in self._pairs.items():
                if a >= b:
                    continue

                rotation_rodrigues = None
                if pair.rotation is not None:
                    rodrigues, _ = cv2.Rodrigues(pair.rotation)
                    rotation_rodrigues = rodrigues.flatten().tolist()

                translation_list = None
                if pair.translation is not None:
                    translation_list = pair.translation.flatten().tolist()

                pair_dict = {
                    "RMSE": pair.error_score,
                    "rotation": rotation_rodrigues,
                    "translation": translation_list,
                }
                pair_dict = {k: v for k, v in pair_dict.items() if v is not None}
                stereo_data[f"stereo_{a}_{b}"] = pair_dict

            _safe_write_toml(stereo_data, path)
        except Exception as e:
            raise PersistenceError(f"Failed to save stereo pairs to {path}: {e}") from e

    def _find_largest_connected_component(self, cam_ids: list[int]) -> set[int]:
        """Finds the largest connected subgraph of cameras."""
        if not self._pairs:
            return set()

        adj = {cam_id: [] for cam_id in cam_ids}
        for cam_id1, cam_id2 in self._pairs.keys():
            if cam_id1 in adj:
                adj[cam_id1].append(cam_id2)

        visited = set()
        largest_component = set()
        for cam_id in cam_ids:
            if cam_id not in visited:
                current_component = set()
                # deque([cam_id]) used in legacy
                from collections import deque

                q = deque([cam_id])
                visited.add(cam_id)
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
