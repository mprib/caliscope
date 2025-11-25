"""
Regression test for StereoPairGraph initialization against gold standard data.

This test validates that the refactored StereoPairGraph produces the same
initial camera extrinsics as the original implementation. It should initially
fail, exposing the discrepancy in the current refactor.
"""

import json
import logging
from pathlib import Path
import re

import numpy as np

from caliscope import __root__
from caliscope.calibration.array_initialization.stereopair_graph import StereoPairGraph
from caliscope.calibration.array_initialization.legacy_stereocalibrator import LegacyStereoCalibrator
from caliscope.cameras.camera_array import CameraArray
from caliscope.configurator import Configurator
from caliscope.post_processing.point_data import ImagePoints

logger = logging.getLogger(__name__)


def load_gold_standard_data() -> tuple[dict, dict]:
    """Load gold standard reference data from JSON files."""
    reference_dir = Path(__root__, "tests", "reference", "stereograph_gold_standard")

    with open(reference_dir / "raw_stereograph.json", "r") as f:
        raw_stereograph = json.load(f)

    with open(reference_dir / "initial_camera_array.json", "r") as f:
        gold_standard_extrinsics = json.load(f)

    return raw_stereograph, gold_standard_extrinsics


def rotation_matrix_to_angle_axis(R: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Convert rotation matrix to angle-axis representation for easier comparison.
    Returns (angle_in_radians, unit_axis_vector)
    """
    # Ensure R is 3x3
    R = R[:3, :3]

    # Compute angle
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
    angle = np.arccos(cos_angle)

    # Compute axis
    if np.isclose(angle, 0):
        return 0.0, np.array([0, 0, 1], dtype=np.float64)

    # Use skew-symmetric part to extract axis
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    axis = axis / (2 * np.sin(angle))

    return angle, axis


def compare_rotations(R_computed: np.ndarray, R_gold: np.ndarray, port: int, label: str) -> dict:
    """
    Compare two rotation matrices and return detailed diagnostics.
    """
    # Compute relative rotation: R_gold^T * R_computed
    R_rel = R_gold.T @ R_computed

    # Convert to angle-axis for intuitive error metric
    angle_error, axis_error = rotation_matrix_to_angle_axis(R_rel)

    # Compute Frobenius norm difference
    frobenius_error = np.linalg.norm(R_computed - R_gold, "fro")

    # Compute per-element differences
    elementwise_diff = R_computed - R_gold
    max_element_error = np.max(np.abs(elementwise_diff))

    return {
        "port": port,
        "label": label,
        "angle_error_rad": angle_error,
        "angle_error_deg": np.degrees(angle_error),
        "axis_error": axis_error,
        "frobenius_error": frobenius_error,
        "max_element_error": max_element_error,
        "R_computed": R_computed,
        "R_gold": R_gold,
    }


def compare_translations(t_computed: np.ndarray, t_gold: np.ndarray, port: int, label: str) -> dict:
    """
    Compare two translation vectors and return detailed diagnostics.
    """
    diff = t_computed - t_gold
    euclidean_error = np.linalg.norm(diff)
    max_component_error = np.max(np.abs(diff))

    # Compute relative error as percentage of gold standard magnitude
    gold_magnitude = np.linalg.norm(t_gold)
    relative_error = (euclidean_error / gold_magnitude * 100) if gold_magnitude > 1e-10 else 0.0

    return {
        "port": port,
        "label": label,
        "euclidean_error": euclidean_error,
        "max_component_error": max_component_error,
        "relative_error_percent": relative_error,
        "t_computed": t_computed,
        "t_gold": t_gold,
    }


def parse_array_string(array_str: str) -> np.ndarray:
    """
    Parse the specific string format used in the gold standard JSON files.
    Handles single-line, multi-line, trailing dots, and scientific notation.
    """
    try:
        # 1. Normalize whitespace (replace newlines/tabs with single spaces)
        clean = re.sub(r"\s+", " ", array_str.strip())

        # 2. Fix the Matrix Format: Replace "] [" with "],[" to separate rows
        # We do this before replacing other spaces to ensure row separation is distinct.
        clean = clean.replace("] [", "],[")

        # 3. Fix formatting specific to numpy strings that breaks JSON:
        # Replace "1." with "1.0" (JSON requires a digit after the dot).
        # Matches a dot followed by space, comma, bracket, or 'e' (scientific notation).
        clean = re.sub(r"\.($|\s|\]|,|e)", r".0\1", clean)

        # 4. Remove padding spaces inside brackets (e.g., "[ 0.1" -> "[0.1")
        clean = clean.replace("[ ", "[")
        clean = clean.replace(" ]", "]")

        # 5. Replace any remaining spaces between numbers with commas
        clean = clean.replace(" ", ",")

        # 6. Parse as JSON
        parsed = json.loads(clean)
        return np.array(parsed, dtype=np.float64)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse array string: {clean}")
        logger.error(f"Original string: {array_str}")
        raise e


def test_stereopair_graph_against_gold_standard():
    """
    Test that StereoPairGraph produces gold-standard initial extrinsics.
    """
    # Load gold standard data
    raw_stereograph, gold_standard_extrinsics = load_gold_standard_data()

    # Load test session data
    version = "larger_calibration_post_monocal"
    session_path = Path(__root__, "tests", "sessions", version)
    config = Configurator(session_path)
    camera_array: CameraArray = config.get_camera_array()

    # Store original camera count
    len(camera_array.cameras)

    # Build StereoPairGraph from gold standard raw data
    logger.info("Building StereoPairGraph from gold standard raw data...")
    gold_stereo_graph = StereoPairGraph.from_legacy_dict(raw_stereograph)

    recording_path = Path(session_path, "calibration", "extrinsic")
    xy_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")
    camera_array = config.get_camera_array()
    config.get_charuco()

    logger.info("Creating stereocalibrator")
    image_points = ImagePoints.from_csv(xy_data_path)
    stereocalibrator = LegacyStereoCalibrator(camera_array, image_points)

    logger.info("Initiating stereocalibration")
    stereo_graph: StereoPairGraph = stereocalibrator.stereo_calibrate_all(boards_sampled=10)

    logger.info("New Stereo Graph Calculated")

    # TODO: Need to do comparisons with assertions here..
    # ... (previous code in test_stereopair_graph_against_gold_standard) ...

    logger.info("New Stereo Graph Calculated")

    # ==============================================================================
    # COMPARE CALCULATED GRAPH AGAINST GOLD STANDARD
    # ==============================================================================

    # 1. Validate Structure (Keys)
    calculated_keys = set(stereo_graph._pairs.keys())
    gold_keys = set(gold_stereo_graph._pairs.keys())

    missing_in_calculated = gold_keys - calculated_keys
    extra_in_calculated = calculated_keys - gold_keys

    assert not missing_in_calculated, f"Missing pairs: {missing_in_calculated}"
    assert not extra_in_calculated, f"Extra pairs: {extra_in_calculated}"

    # 2. Validate Extrinsics
    failures = []

    # TOLERANCES
    # 0.052 rad is approx 3.0 degrees.
    # This allows for drift in 'bridged' pairs while catching wrong orientations.
    ROTATION_TOLERANCE_RAD = 0.052

    # 0.1 units. Assuming standard calibration boards,
    # being off by >10% of a unit is usually a failure.
    TRANSLATION_TOLERANCE = 0.1

    for pair_key in gold_keys:
        port_a, port_b = pair_key
        pair_label = f"{port_a}->{port_b}"

        gold_pair = gold_stereo_graph._pairs[pair_key]
        calc_pair = stereo_graph._pairs[pair_key]

        # --- Compare Rotation ---
        rot_diag = compare_rotations(calc_pair.rotation, gold_pair.rotation, port_a, pair_label)

        # --- Compare Translation ---
        trans_diag = compare_translations(calc_pair.translation, gold_pair.translation, port_a, pair_label)

        # --- Assertions ---
        pair_failed = False
        failure_reasons = []

        if rot_diag["angle_error_rad"] > ROTATION_TOLERANCE_RAD:
            pair_failed = True
            failure_reasons.append(f"Rot Err: {np.degrees(rot_diag['angle_error_rad']):.2f} deg")

        if trans_diag["euclidean_error"] > TRANSLATION_TOLERANCE:
            pair_failed = True
            failure_reasons.append(f"Trans Err: {trans_diag['euclidean_error']:.4f}")

        if pair_failed:
            failures.append(f"{pair_label}: " + ", ".join(failure_reasons))

    # 3. Final Report
    if failures:
        # Sort failures to group related cameras
        failures.sort()

        logger.error(f"{'=' * 60}")
        logger.error(f"REGRESSION TEST FAILED: {len(failures)} pairs exceed tolerance")
        logger.error(f"Tolerances -> Rot: {np.degrees(ROTATION_TOLERANCE_RAD)} deg, Trans: {TRANSLATION_TOLERANCE}")
        logger.error(f"{'=' * 60}")

        for f in failures:
            logger.error(f"  FAILED: {f}")

        raise AssertionError(f"StereoPairGraph regression failed. {len(failures)} pairs mismatch. See logs.")

    logger.info(
        f"SUCCESS: Verified {len(gold_keys)} pairs within {np.degrees(ROTATION_TOLERANCE_RAD):.1f} deg tolerance."
    )


if __name__ == "__main__":
    from caliscope.logger import setup_logging

    setup_logging()
    test_stereopair_graph_against_gold_standard()
