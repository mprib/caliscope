"""
Regression test for StereoPairGraph initialization against gold standard data.

This test validates that the refactored StereoPairGraph produces the same
initial camera extrinsics as the original implementation.
"""

import json
import logging
from pathlib import Path
import re
import numpy as np
import pytest

from caliscope import __root__
from caliscope.calibration.array_initialization.stereopair_graph import StereoPairGraph
from caliscope.calibration.array_initialization.legacy_stereocalibrator import LegacyStereoCalibrator
from caliscope.configurator import Configurator
from caliscope.post_processing.point_data import ImagePoints

logger = logging.getLogger(__name__)

# --- Thresholds ---
# Angle error in radians (approx 0.5 degrees = 0.008 rad)
ROTATION_TOLERANCE_RAD = 1e-4
# Absolute difference in translation units
TRANSLATION_TOLERANCE = 1e-3


def rotation_matrix_to_angle_axis(R: np.ndarray) -> tuple[float, np.ndarray]:
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


def compare_rotations(R_computed: np.ndarray, R_gold: np.ndarray) -> dict:
    """Compare two rotation matrices."""
    # Compute relative rotation: R_gold^T * R_computed
    R_rel = R_gold.T @ R_computed
    angle_error, _ = rotation_matrix_to_angle_axis(R_rel)
    return {"angle_error_rad": angle_error, "angle_error_deg": np.degrees(angle_error)}


def compare_translations(t_computed: np.ndarray, t_gold: np.ndarray) -> dict:
    """Compare two translation vectors."""
    # Flatten just in case shapes are (3,1) vs (3,)
    diff = t_computed.flatten() - t_gold.flatten()
    euclidean_error = np.linalg.norm(diff)
    return {
        "euclidean_error": euclidean_error,
    }


def parse_array_string(array_str: str) -> np.ndarray:
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
        logger.error(f"Failed to parse array string: {clean}")
        raise e


def verify_results(stereo_graph: StereoPairGraph, gold_results: dict):
    """
    Compares the calculated StereoPairGraph against the gold standard dict.
    Collects all errors and raises an assertion at the end.
    """
    failures = []
    checked_count = 0

    # The gold standard uses keys like "stereo_1_10"
    # The new graph uses tuple keys like (1, 10)

    for pair_key, pair_obj in stereo_graph._pairs.items():
        port_a, port_b = pair_key

        # Construct Gold Standard Key
        gold_key = f"stereo_{port_a}_{port_b}"

        if gold_key not in gold_results:
            # It's possible the new graph found connections the old one missed,
            # or the gold standard didn't store the inverse direction.
            # We skip validation for pairs not in gold standard.
            continue

        gold_data = gold_results[gold_key]

        # 1. Extract Data
        # Gold data is a list of lists, need numpy array
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
        if rot_stats["angle_error_rad"] > ROTATION_TOLERANCE_RAD:
            errors.append(f"Rotation error {rot_stats['angle_error_rad']:.6f} rad > {ROTATION_TOLERANCE_RAD}")

        if trans_stats["euclidean_error"] > TRANSLATION_TOLERANCE:
            errors.append(f"Translation error {trans_stats['euclidean_error']:.6f} > {TRANSLATION_TOLERANCE}")

        if errors:
            failures.append({"pair": gold_key, "errors": errors, "stats": {**rot_stats, **trans_stats}})

    # Final Assertion Logic
    if len(failures) > 0:
        error_msg = [f"\nFAILED: {len(failures)} pairs exceeded tolerance."]
        for f in failures:
            error_msg.append(f"  {f['pair']}: {'; '.join(f['errors'])}")

        # Print full details for debugging
        print("\n" + "=" * 60)
        print("DETAILED FAILURE REPORT")
        print("=" * 60)
        for f in failures:
            print(f"Pair: {f['pair']}")
            print(f"  Errors: {f['errors']}")
            print(f"  Angle Error (deg): {f['stats']['angle_error_deg']:.4f}")
            print(f"  Trans Error:       {f['stats']['euclidean_error']:.6f}")
            print("-" * 20)

        pytest.fail("\n".join(error_msg))

    print(f"\nSUCCESS: Verified {checked_count} pairs against gold standard.")


def test_stereopair_graph_against_gold_standard():
    """
    Test that StereoPairGraph produces gold-standard initial extrinsics.
    """
    # 1. Load gold standard data
    reference_dir = Path(__root__, "tests", "reference", "stereograph_gold_standard")

    with open(reference_dir / "main_initial_camera_array.json", "r") as f:
        json.load(f)

    with open(reference_dir / "main_stereocal_all_results.json", "r") as f:
        gold_stereocal_all_results = json.load(f)

    # 2. Load test session data
    version = "larger_calibration_post_monocal"
    session_path = Path(__root__, "tests", "sessions", version)
    config = Configurator(session_path)
    camera_array = config.get_camera_array()

    # ensure camera_array has no extrinsics
    for port, cam in camera_array.cameras.items():
        cam.rotation = None
        cam.translation = None

    # 3. Build stereograph from calibration data
    recording_path = Path(session_path, "calibration", "extrinsic")
    xy_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")

    config.get_charuco()
    logger.info("Creating stereocalibrator")
    image_points = ImagePoints.from_csv(xy_data_path)
    stereocalibrator = LegacyStereoCalibrator(camera_array, image_points)

    logger.info("Initiating stereocalibration")
    # Using the same sampling as presumably used in gold standard
    stereo_graph: StereoPairGraph = stereocalibrator.stereo_calibrate_all(boards_sampled=10)

    # 4. Execute Comparison
    verify_results(stereo_graph, gold_stereocal_all_results)


if __name__ == "__main__":
    # Allow running directly or via pytest
    from caliscope.logger import setup_logging

    setup_logging()
    test_stereopair_graph_against_gold_standard()
