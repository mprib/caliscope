# tests/test_paired_pose_network.py
"""
Regression test for PairedPoseNetwork initialization against gold standard data.

This test validates that the refactored PairedPoseNetwork produces the same
initial camera extrinsics as the original implementation.
"""

import json
import logging
from pathlib import Path
import re
import numpy as np
from numpy.typing import NDArray
import pytest

from caliscope import __root__
from caliscope.core.bootstrap_pose.paired_pose_network import PairedPoseNetwork
from caliscope.core.bootstrap_pose.build_paired_pose_network import build_paired_pose_network
from caliscope.core.bootstrap_pose.stereopairs import StereoPair
from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.point_data import ImagePoints

logger = logging.getLogger(__name__)

# --- Thresholds ---
ROTATION_TOLERANCE_RAD = 0.035
# Absolute difference in translation units
TRANSLATION_TOLERANCE = 0.05  # 5 cm


def rotation_matrix_to_angle_axis(R: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    """
    Convert rotation matrix to angle-axis representation for easier comparison.
    Returns (angle_in_radians, unit_axis_vector)
    """
    R = R[:3, :3]
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if np.isclose(angle, 0):
        return 0.0, np.array([0, 0, 1], dtype=np.float64)

    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return 0.0, np.array([0, 0, 1], dtype=np.float64)

    axis = axis / norm
    return angle, axis


def compare_rotations(R_computed: NDArray[np.float64], R_gold: NDArray[np.float64]) -> dict:
    """Compare two rotation matrices."""
    # Compute relative rotation: R_gold^T * R_computed
    R_rel = R_gold.T @ R_computed
    angle_error, _ = rotation_matrix_to_angle_axis(R_rel)
    return {"angle_error_rad": angle_error, "angle_error_deg": np.degrees(angle_error)}


def compare_translations(t_computed: NDArray[np.float64], t_gold: NDArray[np.float64]) -> dict:
    """Compare two translation vectors."""
    # Flatten just in case shapes are (3,1) vs (3,)
    diff = t_computed.flatten() - t_gold.flatten()
    euclidean_error = np.linalg.norm(diff)
    return {
        "euclidean_error": euclidean_error,
    }


def parse_array_string(array_str: str) -> NDArray[np.float64]:
    """Parse the specific string format used in the gold standard JSON files."""
    try:
        clean = re.sub(r"\s+", " ", array_str.strip())
        clean = clean.replace("] [", "],[")
        clean = re.sub(r"\.($|\s|\]|,|e)", r".0\1", clean)
        clean = clean.replace("[ ", "[")
        clean = clean.replace(" ]", "]")
        clean = clean.replace(" ", ",")
        parsed = json.loads(clean)
        return np.array(parsed, dtype=np.float64)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse array string: {array_str}")
        raise e


def verify_results(stereo_graph: PairedPoseNetwork, gold_results: dict):
    """
    Compares the calculated StereoPairGraph against the gold standard dict.
    Logs the detailed error metrics for EVERY pair to the console.
    """
    failures = []
    checked_count = 0
    missing_in_gold = []

    print("\n" + "=" * 80)
    print(f"{'PAIR':<15} | {'ROT ERR (deg)':<15} | {'TRANS ERR':<15} | {'STATUS':<10}")
    print("-" * 80)

    # The gold standard uses keys like "stereo_1_10"
    # The new graph uses tuple keys like (1, 10)

    for pair_key, pair_obj in stereo_graph._pairs.items():
        cam_id_a, cam_id_b = pair_key

        # Construct Gold Standard Key
        gold_key = f"stereo_{cam_id_a}_{cam_id_b}"

        if gold_key not in gold_results:
            missing_in_gold.append(gold_key)
            continue

        gold_data = gold_results[gold_key]

        # 1. Extract Data
        # Gold data is list of lists, need numpy array
        R_gold = np.array(gold_data["rotation"], dtype=np.float64)
        t_gold = np.array(gold_data["translation"], dtype=np.float64)

        R_new = pair_obj.rotation
        t_new = pair_obj.translation

        # 2. Compare
        rot_stats = compare_rotations(R_new, R_gold)
        trans_stats = compare_translations(t_new, t_gold)

        checked_count += 1

        # 3. Check Thresholds
        errors = []
        is_rot_fail = rot_stats["angle_error_rad"] > ROTATION_TOLERANCE_RAD
        is_trans_fail = trans_stats["euclidean_error"] > TRANSLATION_TOLERANCE

        status = "OK"

        if is_rot_fail:
            errors.append(f"Rot Err {rot_stats['angle_error_rad']:.6f} rad")
            status = "FAIL"

        if is_trans_fail:
            errors.append(f"Trans Err {trans_stats['euclidean_error']:.6f}")
            status = "FAIL"

        # 4. Log the details for this specific pair
        # We use degrees for the log because it is easier for humans to read
        log_line = (
            f"{gold_key:<15} | "
            f"{rot_stats['angle_error_deg']:<15.6f} | "
            f"{trans_stats['euclidean_error']:<15.6f} | "
            f"{status:<10}"
        )
        print(log_line)

        if status == "FAIL":
            failures.append({"pair": gold_key, "errors": errors, "stats": {**rot_stats, **trans_stats}})

    print("-" * 80)

    # Report on pairs found in new graph but missing in gold standard
    if missing_in_gold:
        print(f"NOTICE: {len(missing_in_gold)} pairs found in New Graph but NOT in Gold Standard:")
        print(f"  {', '.join(missing_in_gold)}")
        print("-" * 80)

    # Final Assertion Logic
    if len(failures) > 0:
        error_msg = [f"\nFAILED: {len(failures)} pairs exceeded tolerance."]
        for f in failures:
            error_msg.append(f"  {f['pair']}: {'; '.join(f['errors'])}")

        pytest.fail("\n".join(error_msg))

    print(f"\nSUCCESS: Verified {checked_count} pairs against gold standard.")


def test_stereopair_graph_against_gold_standard():
    """
    Test that StereoPairGraph produces gold-standard initial extrinsics.
    """
    # 1. Load gold standard data
    reference_dir = Path(__root__, "tests", "reference", "stereograph_gold_standard")

    # with open(reference_dir / "main_initial_camera_array.json", "r") as f:
    #     json.load(f)

    with open(reference_dir / "main_stereocal_all_results.json", "r") as f:
        gold_stereocal_all_results = json.load(f)

    # 2. Load test session data
    version = "larger_calibration_post_monocal"
    session_path = Path(__root__, "tests", "sessions", version)
    camera_array = CameraArray.from_toml(session_path / "camera_array.toml")

    # ensure camera_array has no extrinsics
    for cam_id, cam in camera_array.cameras.items():
        cam.rotation = None
        cam.translation = None

    # 3. Build stereograph from calibration data
    recording_path = Path(session_path, "calibration", "extrinsic")
    xy_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")

    logger.info("Creating stereocalibrator")
    image_points = ImagePoints.from_csv(xy_data_path)

    logger.info("Initiating stereocalibration")
    # Using the same sampling as presumably used in gold standard
    paired_pose_network = build_paired_pose_network(image_points, camera_array)
    logger.info("Initializing estimated camera positions based on best daisy-chained stereopairs")
    paired_pose_network.apply_to(camera_array)

    # 4. Execute Comparison
    verify_results(paired_pose_network, gold_stereocal_all_results)


def _rotation_about_z(angle_rad: float) -> NDArray[np.float64]:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def test_apply_to_respects_anchor_cam_zero():
    """Camera 0 passed as explicit anchor must land at the origin.

    The network is built so auto-selection would prefer camera 1 (lowest
    total error), so this fails if anchor_cam=0 is treated as "not provided".
    """
    pair_01 = StereoPair(
        primary_cam_id=0,
        secondary_cam_id=1,
        error_score=1.0,
        rotation=_rotation_about_z(np.radians(10)),
        translation=np.array([1.0, 0.0, 0.0]),
    )
    pair_12 = StereoPair(
        primary_cam_id=1,
        secondary_cam_id=2,
        error_score=1.0,
        rotation=_rotation_about_z(np.radians(-10)),
        translation=np.array([0.0, 1.0, 0.0]),
    )
    network = PairedPoseNetwork.from_raw_estimates({pair_01.pair: pair_01, pair_12.pair: pair_12})
    camera_array = CameraArray(cameras={cam_id: CameraData(cam_id=cam_id, size=(1920, 1080)) for cam_id in (0, 1, 2)})

    network.apply_to(camera_array, anchor_cam=0)

    cam_0 = camera_array.cameras[0]
    assert cam_0.rotation is not None and cam_0.translation is not None
    np.testing.assert_allclose(cam_0.rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(cam_0.translation, np.zeros(3), atol=1e-12)

    # Camera 1 is posed directly from the 0->1 pair
    cam_1 = camera_array.cameras[1]
    assert cam_1.rotation is not None and cam_1.translation is not None
    np.testing.assert_allclose(cam_1.rotation, pair_01.rotation, atol=1e-12)
    np.testing.assert_allclose(cam_1.translation, pair_01.translation, atol=1e-12)


if __name__ == "__main__":
    # Allow running directly or via pytest
    from caliscope.logger import setup_logging

    setup_logging()
    test_stereopair_graph_against_gold_standard()
