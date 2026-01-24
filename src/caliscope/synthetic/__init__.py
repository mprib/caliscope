"""Synthetic data generation for calibration testing."""

from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer, strip_extrinsics
from caliscope.synthetic.coverage import compute_coverage_matrix
from caliscope.synthetic.filter_config import FilterConfig
from caliscope.synthetic.scene import SyntheticScene
from caliscope.synthetic.scenario_config import (
    ScenarioConfig,
    default_ring_scenario,
    sparse_coverage_scenario,
)
from caliscope.synthetic.se3_pose import SE3Pose
from caliscope.synthetic.trajectory import Trajectory

__all__ = [
    "CalibrationObject",
    "CameraSynthesizer",
    "SE3Pose",
    "Trajectory",
    "FilterConfig",
    "SyntheticScene",
    "ScenarioConfig",
    "compute_coverage_matrix",
    "default_ring_scenario",
    "sparse_coverage_scenario",
    "strip_extrinsics",
]
