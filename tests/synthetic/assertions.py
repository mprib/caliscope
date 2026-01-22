"""
Assertion functions for comparing camera poses against ground truth.

Uses geodesic distance on SO(3) for rotation error and Euclidean distance
for translation error (comparing camera positions in world coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from caliscope.cameras.camera_array import CameraArray, CameraData


@dataclass
class PoseError:
    """
    Error metrics between an estimated and ground truth camera pose.

    Attributes:
        rotation_deg: Geodesic distance on SO(3) in degrees.
        translation_mm: Euclidean distance between camera positions in mm.
    """

    rotation_deg: float
    translation_mm: float


def pose_error(estimated: CameraData, ground_truth: CameraData) -> PoseError:
    """
    Compute rotation and translation errors between camera poses.

    Rotation error uses geodesic distance on SO(3):
    - Compute relative rotation R_rel = R_est @ R_gt.T
    - Extract angle from R_rel using Rodrigues representation

    Translation error is Euclidean distance between camera positions:
    - Camera position in world = -R.T @ t
    """
    if estimated.rotation is None or estimated.translation is None:
        raise ValueError(f"Estimated camera {estimated.port} lacks pose")
    if ground_truth.rotation is None or ground_truth.translation is None:
        raise ValueError(f"Ground truth camera {ground_truth.port} lacks pose")

    # Rotation error: geodesic distance on SO(3)
    R_rel: np.ndarray = estimated.rotation @ ground_truth.rotation.T
    rodrigues, _ = cv2.Rodrigues(R_rel)
    rotation_rad = np.linalg.norm(rodrigues)
    rotation_deg = float(np.degrees(rotation_rad))

    # Translation error: Euclidean distance between camera positions
    pos_est = -estimated.rotation.T @ estimated.translation
    pos_gt = -ground_truth.rotation.T @ ground_truth.translation
    translation_mm = float(np.linalg.norm(pos_est - pos_gt))

    return PoseError(rotation_deg=rotation_deg, translation_mm=translation_mm)


def cameras_match_ground_truth(
    actual: CameraArray,
    expected: CameraArray,
    rotation_tol_deg: float = 0.5,
    translation_tol_mm: float = 5.0,
    skip_ports: list[int] | None = None,
) -> tuple[bool, str]:
    """
    Check if optimized cameras match ground truth within tolerances.

    Returns:
        (success, message) tuple where message contains failure details if any.
    """
    if skip_ports is None:
        skip_ports = []

    failures = []

    for port in expected.cameras:
        if port in skip_ports:
            continue

        if port not in actual.cameras:
            failures.append(f"Camera {port}: missing from actual")
            continue

        error = pose_error(actual.cameras[port], expected.cameras[port])

        if error.rotation_deg > rotation_tol_deg:
            failures.append(
                f"Camera {port}: rotation error {error.rotation_deg:.3f} deg > {rotation_tol_deg} deg tolerance"
            )

        if error.translation_mm > translation_tol_mm:
            failures.append(
                f"Camera {port}: translation error {error.translation_mm:.2f} mm > {translation_tol_mm} mm tolerance"
            )

    if failures:
        return False, "\n".join(failures)
    return True, ""


def assert_cameras_moved(
    initial: CameraArray,
    final: CameraArray,
    skip_ports: list[int] | None = None,
    min_movement: float = 1e-6,
) -> None:
    """
    Assert that cameras moved during optimization.

    This catches the critical bug where camera parameters are unpacked
    but never assigned back.
    """
    if skip_ports is None:
        skip_ports = []

    for port in initial.cameras:
        if port in skip_ports:
            continue

        initial_vec = initial.cameras[port].extrinsics_to_vector()
        final_vec = final.cameras[port].extrinsics_to_vector()

        movement = np.linalg.norm(final_vec - initial_vec)

        assert movement > min_movement, (
            f"Camera {port} didn't move during optimization! Parameter change: {movement:.2e}"
        )
