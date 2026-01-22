"""
Synthetic data tests for extrinsic calibration via bundle adjustment.

Run tests:
    uv run pytest tests/synthetic/test_extrinsic_calibration_synthetic.py -v

Visual verification:
    uv run python tests/synthetic/test_extrinsic_calibration_synthetic.py

Theory (ELI5)
─────────────
We inject known noise into perfect synthetic data, then verify bundle adjustment
can recover the original camera positions. It's like a teacher grading their own
answer key - we know exactly what the right answer is.

Why the tolerances are what they are (from covariance propagation theory):
- RMSE should converge to roughly the pixel noise level (χ² statistics)
- Translation error scales linearly with pixel noise: ~15-20x for our geometry
- For pixel_sigma=0.5, expect max translation error of ~7-10mm

The geometry factor (15-20x) comes from: σ_trans ≈ Z²/(f*B) * σ_pixel * k
where Z=depth, f=focal length, B=baseline, and k accounts for multi-camera
coupling and edge effects. See Hartley & Zisserman Ch.18 for derivation.

Gauge Freedom
─────────────
Bundle adjustment has 7 degrees of freedom (3 rotation, 3 translation, 1 scale)
that can't be determined from images alone. We resolve this by calling
align_to_object() which uses known object coordinates (obj_loc_x/y/z) to snap
the optimized result back to the ground truth coordinate frame. This means
ALL cameras are perturbed and optimized - no "gauge reference" camera is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct script execution (debugger, python script.py)
# Pytest handles this automatically, but direct execution needs it
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest  # noqa: E402

from tests.synthetic.assertions import (  # noqa: E402
    assert_cameras_moved,
    cameras_match_ground_truth,
)
from tests.synthetic.test_cases import (  # noqa: E402
    ExtrinsicCalibrationTestCase,
    create_extrinsic_calibration_test_case,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_case() -> ExtrinsicCalibrationTestCase:
    """
    Create test case once for all tests in module.

    Module scope is safe because ExtrinsicCalibrationTestCase is frozen/immutable.
    """
    # Use defaults: rotation_sigma=0.10 (~5.7 deg), translation_sigma=100.0 mm
    return create_extrinsic_calibration_test_case(
        seed=42,
        n_frames=20,
        pixel_sigma=0.5,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extrinsic_calibration_cameras_moved(test_case: ExtrinsicCalibrationTestCase):
    """
    CRITICAL: Verify that all cameras moved during optimization.

    This test catches the bug where camera parameters are unpacked from the
    optimization vector but never assigned back, causing only 3D points to
    be optimized while cameras stay frozen.
    """
    assert_cameras_moved(
        initial=test_case.noisy_input.cameras,
        final=test_case.optimized_bundle.camera_array,
        skip_ports=[],  # All cameras should move
    )


def test_extrinsic_calibration_cameras_converged(test_case: ExtrinsicCalibrationTestCase):
    """
    Verify that optimized cameras are close to ground truth.

    Tolerances are derived from theory, not arbitrary:
    - Rotation: < 0.5 degrees (empirically stable across seeds)
    - Translation: < 10 mm = 20 * pixel_sigma (geometry factor ~15-20x)

    The geometry factor comes from σ_trans ≈ Z²/(f*B) * σ_pixel * k, where
    k accounts for camera pose coupling and max vs mean (4σ bound).
    For our setup (Z=2000mm, f=800px, B~2000mm), this gives ~15-20 mm/pixel.
    """
    # Theory-based tolerances for pixel_sigma=0.5
    PIXEL_SIGMA = 0.5
    GEOMETRY_FACTOR = 20.0  # mm per pixel of noise (conservative)

    success, errors = cameras_match_ground_truth(
        actual=test_case.optimized_bundle.camera_array,
        expected=test_case.ground_truth.cameras,
        rotation_tol_deg=0.5,
        translation_tol_mm=GEOMETRY_FACTOR * PIXEL_SIGMA,  # 10mm for 0.5px noise
        skip_ports=[],  # All cameras should converge
    )
    assert success, f"Camera recovery failed:\n{errors}"


def test_extrinsic_calibration_error_decreased(test_case: ExtrinsicCalibrationTestCase):
    """
    Verify that pose errors decreased for all cameras.

    This is a weaker assertion than convergence to ground truth, but it
    catches cases where optimization made things worse.
    """
    for port in test_case.ground_truth.cameras.cameras:
        initial = test_case.initial_pose_errors[port]
        final = test_case.final_pose_errors[port]

        assert final.rotation_deg < initial.rotation_deg, (
            f"Camera {port}: rotation error increased from {initial.rotation_deg:.3f} to {final.rotation_deg:.3f} deg"
        )
        assert final.translation_mm < initial.translation_mm, (
            f"Camera {port}: translation error increased from "
            f"{initial.translation_mm:.2f} to {final.translation_mm:.2f} mm"
        )


def test_extrinsic_calibration_significant_improvement(test_case: ExtrinsicCalibrationTestCase):
    """
    Verify that optimization achieved significant error reduction (>80%).

    This catches cases where optimization converged to a local minimum
    without meaningful improvement.
    """
    for port in test_case.ground_truth.cameras.cameras:
        initial = test_case.initial_pose_errors[port]
        final = test_case.final_pose_errors[port]

        rot_improvement = (initial.rotation_deg - final.rotation_deg) / initial.rotation_deg
        trans_improvement = (initial.translation_mm - final.translation_mm) / initial.translation_mm

        assert rot_improvement > 0.8, (
            f"Camera {port}: rotation only improved by {rot_improvement * 100:.1f}% "
            f"({initial.rotation_deg:.3f} -> {final.rotation_deg:.3f} deg)"
        )
        assert trans_improvement > 0.8, (
            f"Camera {port}: translation only improved by {trans_improvement * 100:.1f}% "
            f"({initial.translation_mm:.2f} -> {final.translation_mm:.2f} mm)"
        )


# ---------------------------------------------------------------------------
# Visual Verification
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    from tests.synthetic.widgets.storyboard import CalibrationStoryboardWidget

    print("Creating test case...")
    # Use defaults: rotation_sigma=0.10 (~5.7 deg), translation_sigma=100.0 mm
    visual_test_case = create_extrinsic_calibration_test_case(
        seed=42,
        n_frames=20,
        pixel_sigma=0.5,
    )

    print("Launching visualization...")
    app = QApplication(sys.argv)
    widget = CalibrationStoryboardWidget(visual_test_case)
    widget.setWindowTitle("Extrinsic Calibration - Synthetic Data Verification")
    widget.resize(1800, 700)
    widget.show()
    sys.exit(app.exec())
