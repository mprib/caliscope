"""Configuration for a complete synthetic calibration scenario."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
from caliscope.synthetic.filter_config import CameraOcclusion, FilterConfig
from caliscope.synthetic.scene import SyntheticScene
from caliscope.synthetic.trajectory import Trajectory


@dataclass
class ScenarioConfig:
    """Configuration for a synthetic calibration scenario.

    Use factory functions (default_ring_scenario, sparse_coverage_scenario, etc.)
    for common configurations. Call build_scene() to produce an executable SyntheticScene.
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

    def build_scene(self) -> SyntheticScene:
        """Construct the SyntheticScene from this config."""
        # Build camera rig using CameraSynthesizer
        if self.rig_type == "ring":
            camera_array = (
                CameraSynthesizer()
                .add_ring(
                    n=self.rig_params["n_cameras"],
                    radius_mm=self.rig_params["radius_mm"],
                    height_mm=self.rig_params.get("height_mm", 0.0),
                    facing=self.rig_params.get("facing", "inward"),
                )
                .build()
            )
        elif self.rig_type == "linear":
            camera_array = (
                CameraSynthesizer()
                .add_line(
                    n=self.rig_params["n_cameras"],
                    spacing_mm=self.rig_params["spacing_mm"],
                    curvature=self.rig_params.get("curvature", 0.0),
                )
                .build()
            )
        elif self.rig_type == "nested_rings":
            camera_array = (
                CameraSynthesizer()
                .add_ring(
                    n=self.rig_params["inner_n"],
                    radius_mm=self.rig_params["inner_radius_mm"],
                    height_mm=self.rig_params.get("inner_height_mm", 0.0),
                    facing="outward",
                )
                .add_ring(
                    n=self.rig_params["outer_n"],
                    radius_mm=self.rig_params["outer_radius_mm"],
                    height_mm=self.rig_params.get("outer_height_mm", 500.0),
                    facing="inward",
                )
                .build()
            )
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
