"""Serializable configuration for a complete synthetic calibration scenario."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import rtoml

from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_rigs import linear_rig, nested_rings_rig, ring_rig
from caliscope.synthetic.filter_config import CameraOcclusion, FilterConfig
from caliscope.synthetic.scene import SyntheticScene
from caliscope.synthetic.trajectory import Trajectory


@dataclass
class ScenarioConfig:
    """Serializable configuration for a synthetic scenario.

    Can be exported from Explorer GUI and loaded by pytest for automated testing.
    Designed for TOML serialization with human-readable structure.
    """

    # Camera rig configuration
    rig_type: Literal["ring", "linear", "nested_rings"]
    rig_params: dict[str, Any]

    # Trajectory configuration
    trajectory_type: Literal["orbital", "linear", "stationary"]
    trajectory_params: dict[str, Any]

    # Calibration object configuration
    object_type: Literal["planar_grid"]
    object_params: dict[str, Any]

    # Noise and filtering
    pixel_noise_sigma: float = 0.5
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    random_seed: int = 42

    # Metadata
    name: str = "unnamed"
    description: str = ""

    def to_toml(self) -> str:
        """Serialize to TOML string."""
        data = {
            "metadata": {
                "name": self.name,
                "description": self.description,
            },
            "camera_rig": {
                "type": self.rig_type,
                **self.rig_params,
            },
            "trajectory": {
                "type": self.trajectory_type,
                **self.trajectory_params,
            },
            "calibration_object": {
                "type": self.object_type,
                **self.object_params,
            },
            "noise": {
                "pixel_sigma": self.pixel_noise_sigma,
                "seed": self.random_seed,
            },
            "filter": self.filter_config.to_dict(),
        }
        return rtoml.dumps(data)

    @classmethod
    def from_toml(cls, toml_str: str) -> ScenarioConfig:
        """Deserialize from TOML string."""
        data = rtoml.loads(toml_str)

        metadata = data.get("metadata", {})
        rig = data["camera_rig"]
        traj = data["trajectory"]
        obj = data["calibration_object"]
        noise = data.get("noise", {})
        filter_data = data.get("filter", {})

        # Extract type and remaining params
        rig_type = rig.pop("type")
        traj_type = traj.pop("type")
        obj_type = obj.pop("type")

        return cls(
            rig_type=rig_type,
            rig_params=rig,
            trajectory_type=traj_type,
            trajectory_params=traj,
            object_type=obj_type,
            object_params=obj,
            pixel_noise_sigma=noise.get("pixel_sigma", 0.5),
            random_seed=noise.get("seed", 42),
            filter_config=FilterConfig.from_dict(filter_data),
            name=metadata.get("name", "unnamed"),
            description=metadata.get("description", ""),
        )

    def build_scene(self) -> SyntheticScene:
        """Construct the SyntheticScene from this config."""
        # Build camera rig
        if self.rig_type == "ring":
            camera_array = ring_rig(**self.rig_params)
        elif self.rig_type == "linear":
            camera_array = linear_rig(**self.rig_params)
        elif self.rig_type == "nested_rings":
            camera_array = nested_rings_rig(**self.rig_params)
        else:
            raise ValueError(f"Unknown rig type: {self.rig_type}")

        # Build trajectory
        if self.trajectory_type == "orbital":
            trajectory = Trajectory.orbital(**self.trajectory_params)
        elif self.trajectory_type == "linear":
            # Convert list to numpy array for start/end
            params = self.trajectory_params.copy()
            if "start" in params:
                params["start"] = np.array(params["start"], dtype=np.float64)
            if "end" in params:
                params["end"] = np.array(params["end"], dtype=np.float64)
            trajectory = Trajectory.linear(**params)
        elif self.trajectory_type == "stationary":
            trajectory = Trajectory.stationary(**self.trajectory_params)
        else:
            raise ValueError(f"Unknown trajectory type: {self.trajectory_type}")

        # Build calibration object
        if self.object_type == "planar_grid":
            calibration_object = CalibrationObject.planar_grid(**self.object_params)
        else:
            raise ValueError(f"Unknown object type: {self.object_type}")

        return SyntheticScene(
            camera_array=camera_array,
            calibration_object=calibration_object,
            trajectory=trajectory,
            pixel_noise_sigma=self.pixel_noise_sigma,
            random_seed=self.random_seed,
        )


# Factory functions for common scenarios
def default_ring_scenario() -> ScenarioConfig:
    """4 cameras in a ring, orbital trajectory, 5x7 grid."""
    return ScenarioConfig(
        rig_type="ring",
        rig_params={"n_cameras": 4, "radius_mm": 2000.0, "height_mm": 500.0},
        trajectory_type="orbital",
        trajectory_params={
            "n_frames": 20,
            "radius_mm": 200.0,
            "arc_extent_deg": 360.0,
            "tumble_rate": 1.0,
        },
        object_type="planar_grid",
        object_params={"rows": 5, "cols": 7, "spacing_mm": 50.0},
        name="Default Ring",
        description="Standard 4-camera ring with full orbital trajectory",
    )


def sparse_coverage_scenario() -> ScenarioConfig:
    """4 cameras, partial arc (cameras don't all see same frames)."""
    return ScenarioConfig(
        rig_type="ring",
        rig_params={"n_cameras": 4, "radius_mm": 2000.0, "height_mm": 500.0},
        trajectory_type="orbital",
        trajectory_params={
            "n_frames": 20,
            "radius_mm": 400.0,  # Larger radius = less overlap
            "arc_extent_deg": 180.0,  # Half orbit
            "tumble_rate": 0.5,
        },
        object_type="planar_grid",
        object_params={"rows": 5, "cols": 7, "spacing_mm": 50.0},
        name="Sparse Coverage",
        description="Half orbit - tests cameras with limited shared visibility",
    )


def occluded_camera_scenario() -> ScenarioConfig:
    """One camera loses 50% of shared observations (simulates partial tracking failure)."""
    return ScenarioConfig(
        rig_type="ring",
        rig_params={"n_cameras": 4, "radius_mm": 2000.0, "height_mm": 500.0},
        trajectory_type="orbital",
        trajectory_params={
            "n_frames": 20,
            "radius_mm": 200.0,
            "arc_extent_deg": 360.0,
            "tumble_rate": 1.0,
        },
        object_type="planar_grid",
        object_params={"rows": 5, "cols": 7, "spacing_mm": 50.0},
        filter_config=FilterConfig(camera_occlusions=(CameraOcclusion(camera_port=0, dropout_fraction=0.5),)),
        name="Occluded Camera",
        description="Camera 0 loses 50% of all shared observations",
    )
