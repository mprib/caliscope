"""Declarative filter specification for synthetic observation filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from caliscope.core.point_data import ImagePoints


@dataclass(frozen=True)
class CameraOcclusion:
    """Camera-specific observation dropout.

    Unlike killed_linkages (which removes shared observations from BOTH cameras),
    this only removes observations from the specified camera_port.

    Example: Camera 1 loses 30% of observations it shares with camera 2:
        CameraOcclusion(camera_port=1, dropout_fraction=0.3, target_cameras=(2,))
    """

    camera_port: int  # The camera that loses observations
    dropout_fraction: float  # 0.0 to 1.0 - fraction of observations to drop
    target_cameras: tuple[int, ...] = ()  # Empty = all cameras, else specific cameras
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not (0.0 <= self.dropout_fraction <= 1.0):
            raise ValueError(f"dropout_fraction must be in [0, 1], got {self.dropout_fraction}")


@dataclass(frozen=True)
class FilterConfig:
    """Declarative filter specification for synthetic scene observations.

    Filters are applied to ImagePoints to simulate various edge cases:
    - Camera failures (dropped_cameras)
    - Weak inter-camera linkages (killed_linkages)
    - Camera-specific occlusions (camera_occlusions)
    - Temporal gaps (dropped_frame_ranges)
    - Random detection failures (random_dropout_fraction)

    All attributes are immutable tuples for hashing and serialization.
    """

    # Camera-level: remove all observations from these cameras
    dropped_cameras: tuple[int, ...] = ()

    # Linkage-level: sever shared observations between camera pairs
    # Each tuple (a, b) removes points seen by BOTH cameras a and b
    # This breaks the pose network edge between them
    killed_linkages: tuple[tuple[int, int], ...] = ()

    # Camera-specific occlusion: one camera loses observations (not shared with others)
    camera_occlusions: tuple[CameraOcclusion, ...] = ()

    # Frame-level: drop observations in frame ranges (start, end inclusive)
    dropped_frame_ranges: tuple[tuple[int, int], ...] = ()

    # Random degradation
    random_dropout_fraction: float = 0.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate filter configuration."""
        if not (0.0 <= self.random_dropout_fraction < 1.0):
            raise ValueError(f"random_dropout_fraction must be in [0, 1), got {self.random_dropout_fraction}")

        for a, b in self.killed_linkages:
            if a == b:
                raise ValueError(f"Cannot kill linkage with same camera: ({a}, {b})")

    def with_killed_linkage(self, cam_a: int, cam_b: int) -> FilterConfig:
        """Return new FilterConfig with an additional killed linkage.

        Order doesn't matter - (a, b) and (b, a) have the same effect.
        Normalizes to (min, max) for consistency.
        """
        normalized = (min(cam_a, cam_b), max(cam_a, cam_b))

        # Skip if already killed
        if normalized in self.killed_linkages:
            return self

        new_linkages = self.killed_linkages + (normalized,)

        return FilterConfig(
            dropped_cameras=self.dropped_cameras,
            killed_linkages=new_linkages,
            camera_occlusions=self.camera_occlusions,
            dropped_frame_ranges=self.dropped_frame_ranges,
            random_dropout_fraction=self.random_dropout_fraction,
            random_seed=self.random_seed,
        )

    def without_killed_linkage(self, cam_a: int, cam_b: int) -> FilterConfig:
        """Return new FilterConfig with a killed linkage restored."""
        normalized = (min(cam_a, cam_b), max(cam_a, cam_b))

        new_linkages = tuple(link for link in self.killed_linkages if link != normalized)

        return FilterConfig(
            dropped_cameras=self.dropped_cameras,
            killed_linkages=new_linkages,
            camera_occlusions=self.camera_occlusions,
            dropped_frame_ranges=self.dropped_frame_ranges,
            random_dropout_fraction=self.random_dropout_fraction,
            random_seed=self.random_seed,
        )

    def apply(self, image_points: ImagePoints) -> ImagePoints:
        """Apply all filters to image points, returning a filtered copy.

        Filters are applied in order:
        1. Drop cameras
        2. Kill linkages
        3. Camera occlusions
        4. Drop frame ranges
        5. Random dropout

        Args:
            image_points: Source ImagePoints (not modified)

        Returns:
            New ImagePoints with observations removed per filter config
        """
        df = image_points.df.copy()

        # 1. Drop cameras
        if self.dropped_cameras:
            df = df[~df["port"].isin(self.dropped_cameras)]

        # 2. Kill linkages (remove shared observations)
        for cam_a, cam_b in self.killed_linkages:
            df = self._kill_linkage(df, cam_a, cam_b)

        # 3. Camera occlusions (one camera loses observations)
        df = self._apply_camera_occlusions(df)

        # 4. Drop frame ranges
        for start, end in self.dropped_frame_ranges:
            df = df[(df["sync_index"] < start) | (df["sync_index"] > end)]

        # 5. Random dropout
        if self.random_dropout_fraction > 0:
            rng = np.random.default_rng(self.random_seed)
            keep_mask = rng.random(len(df)) > self.random_dropout_fraction
            df = df[keep_mask]

        return ImagePoints(df)

    @staticmethod
    def _kill_linkage(df: pd.DataFrame, cam_a: int, cam_b: int) -> pd.DataFrame:
        """Remove points that are seen by BOTH cameras in a pair.

        For each (sync_index, point_id), if observed by both cam_a and cam_b,
        remove ALL observations of that point at that frame from those cameras.
        This breaks the stereo constraint between them.
        """
        # Find points seen by cam_a
        seen_by_a = set(
            zip(
                df[df["port"] == cam_a]["sync_index"],
                df[df["port"] == cam_a]["point_id"],
            )
        )

        # Find points seen by cam_b
        seen_by_b = set(
            zip(
                df[df["port"] == cam_b]["sync_index"],
                df[df["port"] == cam_b]["point_id"],
            )
        )

        # Shared points
        shared = seen_by_a & seen_by_b

        if not shared:
            return df

        # Create mask to exclude shared observations from these two cameras
        shared_set = set(shared)

        def should_keep(row: pd.Series[Any]) -> bool:
            key = (row["sync_index"], row["point_id"])
            if key in shared_set and row["port"] in (cam_a, cam_b):
                return False
            return True

        # Pandas stubs don't properly type axis=1 apply operations
        keep_mask = df.apply(should_keep, axis=1)  # type: ignore[arg-type]
        return df[keep_mask]

    def _apply_camera_occlusions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply camera-specific occlusions.

        For each CameraOcclusion, randomly drops observations from the specified
        camera_port that are shared with target_cameras (or all cameras if empty).
        Unlike killed_linkages, this only affects the occluded camera, not its peers.
        """
        if not self.camera_occlusions:
            return df

        for occlusion in self.camera_occlusions:
            # Get observations from the occluded camera
            occluded_cam_obs = df[df["port"] == occlusion.camera_port]
            if occluded_cam_obs.empty:
                continue

            # Find observations from target cameras (or all other cameras if empty)
            if occlusion.target_cameras:
                target_df = df[df["port"].isin(occlusion.target_cameras)]
            else:
                target_df = df[df["port"] != occlusion.camera_port]

            if target_df.empty:
                continue

            # Find shared observations (sync_index, point_id pairs)
            occluded_points = set(
                zip(
                    occluded_cam_obs["sync_index"],
                    occluded_cam_obs["point_id"],
                )
            )
            target_points = set(
                zip(
                    target_df["sync_index"],
                    target_df["point_id"],
                )
            )
            shared_points = occluded_points & target_points

            if not shared_points:
                continue

            # Randomly select dropout_fraction of shared points to drop
            rng = np.random.default_rng(occlusion.random_seed)
            shared_list = list(shared_points)
            n_to_drop = int(len(shared_list) * occlusion.dropout_fraction)

            if n_to_drop == 0:
                continue

            dropped_points = set(
                rng.choice(
                    len(shared_list),
                    size=n_to_drop,
                    replace=False,
                )
            )
            points_to_drop = {shared_list[i] for i in dropped_points}

            # Remove only from the occluded camera
            def should_keep(row: pd.Series[Any]) -> bool:
                if row["port"] != occlusion.camera_port:
                    return True
                key = (row["sync_index"], row["point_id"])
                return key not in points_to_drop

            # Pandas stubs don't properly type axis=1 apply operations
            keep_mask = df.apply(should_keep, axis=1)  # type: ignore[arg-type]
            df = df[keep_mask]

        return df

    def to_dict(self) -> dict:
        """Serialize to dictionary for TOML export."""
        return {
            "dropped_cameras": list(self.dropped_cameras),
            "killed_linkages": [list(link) for link in self.killed_linkages],
            "camera_occlusions": [
                {
                    "camera_port": occ.camera_port,
                    "dropout_fraction": occ.dropout_fraction,
                    "target_cameras": list(occ.target_cameras),
                    "random_seed": occ.random_seed,
                }
                for occ in self.camera_occlusions
            ],
            "dropped_frame_ranges": [list(r) for r in self.dropped_frame_ranges],
            "random_dropout_fraction": self.random_dropout_fraction,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FilterConfig:
        """Deserialize from dictionary."""
        occlusion_data = data.get("camera_occlusions", [])
        camera_occlusions = tuple(
            CameraOcclusion(
                camera_port=occ["camera_port"],
                dropout_fraction=occ["dropout_fraction"],
                target_cameras=tuple(occ.get("target_cameras", ())),
                random_seed=occ.get("random_seed", 42),
            )
            for occ in occlusion_data
        )

        return cls(
            dropped_cameras=tuple(data.get("dropped_cameras", [])),
            killed_linkages=tuple(tuple(link) for link in data.get("killed_linkages", [])),
            camera_occlusions=camera_occlusions,
            dropped_frame_ranges=tuple(tuple(r) for r in data.get("dropped_frame_ranges", [])),
            random_dropout_fraction=data.get("random_dropout_fraction", 0.0),
            random_seed=data.get("random_seed", 42),
        )
