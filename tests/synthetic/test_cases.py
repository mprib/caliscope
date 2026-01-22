"""
Test case dataclasses for extrinsic calibration testing.

Provides frozen, immutable containers that hold the complete state of a calibration
test: ground truth, noisy input, and optimization result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints
from caliscope.core.point_data_bundle import PointDataBundle
from tests.synthetic.assertions import PoseError, pose_error
from tests.synthetic.generators.ground_truth import SyntheticGroundTruth, create_four_camera_ring


@dataclass(frozen=True)
class NoisyCalibrationInput:
    """
    The input fed to the optimizer: perturbed cameras + noisy image points.

    This represents what a real calibration system would receive:
    - Cameras with approximate initial extrinsics (from stereo bootstrapping)
    - 2D image observations with detection noise

    Attributes:
        cameras: CameraArray with perturbed extrinsics
        image_points: ImagePoints with Gaussian noise added
    """

    cameras: CameraArray
    image_points: ImagePoints


@dataclass(frozen=True)
class ExtrinsicCalibrationTestCase:
    """
    Complete test case for extrinsic calibration via bundle adjustment.

    Contains all three stages of calibration testing:
    1. Ground truth (the oracle)
    2. Noisy input (what optimizer receives)
    3. Optimized result (what optimizer produces)

    Used by both pytest assertions and visual verification widgets.

    Attributes:
        ground_truth: Perfect cameras, world points, and image projections
        noisy_input: Perturbed cameras and noisy image observations
        optimized_bundle: Result of running PointDataBundle.optimize()
    """

    ground_truth: SyntheticGroundTruth
    noisy_input: NoisyCalibrationInput
    optimized_bundle: PointDataBundle

    # Computed properties use field(init=False) pattern for frozen dataclass
    _initial_errors: dict[int, PoseError] = field(init=False, repr=False)
    _final_errors: dict[int, PoseError] = field(init=False, repr=False)

    def __post_init__(self):
        """Compute pose errors on construction."""
        initial = {}
        final = {}

        for port in self.ground_truth.cameras.cameras:
            gt_camera = self.ground_truth.cameras.cameras[port]
            noisy_camera = self.noisy_input.cameras.cameras[port]
            opt_camera = self.optimized_bundle.camera_array.cameras[port]

            initial[port] = pose_error(noisy_camera, gt_camera)
            final[port] = pose_error(opt_camera, gt_camera)

        object.__setattr__(self, "_initial_errors", initial)
        object.__setattr__(self, "_final_errors", final)

    @property
    def initial_pose_errors(self) -> dict[int, PoseError]:
        """Per-camera pose error between noisy_input and ground_truth."""
        return self._initial_errors

    @property
    def final_pose_errors(self) -> dict[int, PoseError]:
        """Per-camera pose error between optimized_bundle and ground_truth."""
        return self._final_errors

    @property
    def sync_indices(self) -> list[int]:
        """Available frame indices for visualization slider."""
        return sorted(self.ground_truth.world_points.df["sync_index"].unique().tolist())

    @property
    def initial_reprojection_rmse(self) -> float:
        """RMSE for noisy input state (perturbed cameras + noisy observations)."""
        # Create bundle from noisy state to compute reprojection error
        # Note: This is computed fresh each access - could cache in __post_init__ if needed
        triangulated_world_points = self.noisy_input.image_points.triangulate(self.noisy_input.cameras)
        initial_bundle = PointDataBundle(
            camera_array=self.noisy_input.cameras,
            image_points=self.noisy_input.image_points,
            world_points=triangulated_world_points,
        )
        return initial_bundle.reprojection_report.overall_rmse

    @property
    def final_reprojection_rmse(self) -> float:
        """RMSE for optimized state."""
        return self.optimized_bundle.reprojection_report.overall_rmse


def create_extrinsic_calibration_test_case(
    seed: int = 42,
    n_frames: int = 20,
    rotation_sigma: float = 0.10,
    translation_sigma: float = 100.0,
    pixel_sigma: float = 0.5,
    optimize_ftol: float = 1e-8,
    optimize_max_nfev: int | None = None,
) -> ExtrinsicCalibrationTestCase:
    """
    Create a complete test case by generating ground truth, adding noise, and running optimization.

    This is the single source of truth for both pytest assertions and visual verification.

    Args:
        seed: Random seed for reproducibility
        n_frames: Number of temporal frames in the synthetic trajectory
        rotation_sigma: Camera rotation perturbation in radians (default: 0.10 ~ 5.7 degrees)
        translation_sigma: Camera translation perturbation in mm (default: 100.0 mm)
        pixel_sigma: Image point noise in pixels (default: 0.5 pixels)
        optimize_ftol: Tolerance for optimization convergence
        optimize_max_nfev: Maximum function evaluations (None = no limit)

    Returns:
        ExtrinsicCalibrationTestCase with ground truth, noisy input, and optimization result

    Note:
        All cameras are perturbed. Gauge freedom (7 DOF) is resolved by align_to_object()
        which aligns the optimized bundle to known object coordinates.
    """

    rng = np.random.default_rng(seed)

    # Step 1: Generate ground truth
    ground_truth = create_four_camera_ring(seed=seed, n_frames=n_frames)

    # Step 2: Create perturbed cameras (all cameras perturbed - gauge resolved by align_to_object)
    perturbed_cameras = ground_truth.with_camera_perturbation(
        rotation_sigma=rotation_sigma,
        translation_sigma=translation_sigma,
        rng=rng,
        fixed_ports=[],  # Perturb all cameras
    )

    # Step 3: Create noisy image observations
    noisy_image_points = ground_truth.with_image_noise(
        pixel_sigma=pixel_sigma,
        rng=rng,
    )

    # Step 4: Triangulate world points using perturbed cameras
    # ImagePoints.triangulate() returns WorldPoints using the camera array's poses
    triangulated_world_points = noisy_image_points.triangulate(perturbed_cameras)

    # Step 5: Create initial PointDataBundle
    initial_bundle = PointDataBundle(
        camera_array=perturbed_cameras,
        image_points=noisy_image_points,
        world_points=triangulated_world_points,
    )

    # Step 6: Run optimization
    optimized_bundle = initial_bundle.optimize(
        ftol=optimize_ftol,
        max_nfev=optimize_max_nfev,
        verbose=0,
    )

    # Step 7: Align back to ground truth frame using known object coordinates
    # Bundle adjustment has gauge freedom (7 DOF: rotation, translation, scale).
    # The optimizer finds a valid solution but may drift to a different coordinate frame.
    # align_to_object() uses the known obj_loc_x/y/z values to snap back to ground truth.
    optimized_bundle = optimized_bundle.align_to_object(sync_index=0)

    # Step 8: Assemble test case
    noisy_input = NoisyCalibrationInput(
        cameras=perturbed_cameras,
        image_points=noisy_image_points,
    )

    return ExtrinsicCalibrationTestCase(
        ground_truth=ground_truth,
        noisy_input=noisy_input,
        optimized_bundle=optimized_bundle,
    )
