"""Synthetic data generation for calibration testing."""

from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import (
    IDEAL,
    MACHINE_VISION,
    WEBCAM,
    CameraSynthesizer,
    IntrinsicPerturbation,
    LensProfile,
    perturb_intrinsics,
    strip_extrinsics,
    strip_intrinsics,
)
from caliscope.core.coverage_analysis import compute_coverage_matrix
from caliscope.synthetic.filter_config import FilterConfig
from caliscope.synthetic.outliers import OutlierConfig, inject_outliers
from caliscope.synthetic.fixture_repository import (
    SyntheticFixture,
    SyntheticFixtureRepository,
    load_fixture,
    save_fixture,
)
from caliscope.synthetic.synthetic_scene import SceneObject, SyntheticScene
from caliscope.synthetic.scene_factories import (
    aruco_multi_object_scene,
    aruco_scene,
    chain_scene,
    cheirality_demo_scene,
    default_ring_scene,
    large_ring_scene,
    narrow_baseline_scene,
    outlier_scene,
    quick_test_scene,
    sparse_coverage_scene,
    visibility_culling_scene,
)
from caliscope.synthetic.scene_factories import box_target_scene, charuco_target_scene, machine_vision_scene
from caliscope.synthetic.se3_pose import SE3Pose
from caliscope.synthetic.target_factories import (
    aruco_marker,
    box_target,
    charuco_board,
    double_sided_charuco_board,
)
from caliscope.synthetic.trajectory import Trajectory

__all__ = [
    "CalibrationObject",
    "CameraSynthesizer",
    "SE3Pose",
    "Trajectory",
    "FilterConfig",
    "SceneObject",
    "SyntheticScene",
    "compute_coverage_matrix",
    "aruco_marker",
    "aruco_multi_object_scene",
    "aruco_scene",
    "box_target",
    "box_target_scene",
    "charuco_board",
    "charuco_target_scene",
    "IDEAL",
    "IntrinsicPerturbation",
    "LensProfile",
    "MACHINE_VISION",
    "machine_vision_scene",
    "chain_scene",
    "cheirality_demo_scene",
    "default_ring_scene",
    "large_ring_scene",
    "narrow_baseline_scene",
    "outlier_scene",
    "sparse_coverage_scene",
    "quick_test_scene",
    "visibility_culling_scene",
    "OutlierConfig",
    "inject_outliers",
    "double_sided_charuco_board",
    "SyntheticFixture",
    "SyntheticFixtureRepository",
    "save_fixture",
    "load_fixture",
    "perturb_intrinsics",
    "strip_extrinsics",
    "strip_intrinsics",
    "WEBCAM",
]
