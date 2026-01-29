"""Synthetic data generation for calibration testing."""

from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer, strip_extrinsics
from caliscope.core.coverage_analysis import compute_coverage_matrix
from caliscope.synthetic.filter_config import FilterConfig
from caliscope.synthetic.fixture_repository import (
    SyntheticFixture,
    SyntheticFixtureRepository,
    load_fixture,
    save_fixture,
)
from caliscope.synthetic.synthetic_scene import SyntheticScene
from caliscope.synthetic.scene_factories import (
    default_ring_scene,
    quick_test_scene,
    sparse_coverage_scene,
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
    "compute_coverage_matrix",
    "default_ring_scene",
    "sparse_coverage_scene",
    "quick_test_scene",
    "SyntheticFixture",
    "SyntheticFixtureRepository",
    "save_fixture",
    "load_fixture",
    "strip_extrinsics",
]
