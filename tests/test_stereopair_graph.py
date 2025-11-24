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
from caliscope.cameras.camera_array import CameraArray
from caliscope.configurator import Configurator

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
    original_camera_count = len(camera_array.cameras)

    # Build StereoPairGraph from gold standard raw data
    logger.info("Building StereoPairGraph from gold standard raw data...")
    stereo_graph = StereoPairGraph.from_legacy_dict(raw_stereograph)

    logger.info(f"Graph contains {len(stereo_graph._pairs)} directed pairs")

    # Apply graph to compute extrinsics
    logger.info("Applying graph to compute camera extrinsics...")
    stereo_graph.apply_to(camera_array)

    # Verify all cameras are posed (except possibly ignored ones)
    posed_count = len(camera_array.posed_cameras)
    logger.info(f"Cameras posed: {posed_count}/{original_camera_count}")

    # Perform detailed comparison
    print("\n" + "=" * 80)
    print("STEREOPAIR GRAPH GOLD STANDARD COMPARISON")
    print("=" * 80)

    rotation_results = []
    translation_results = []
    all_ports = sorted(camera_array.cameras.keys())

    for port in all_ports:
        cam = camera_array.cameras[port]
        if cam.rotation is None or cam.translation is None:
            logger.warning(f"Camera {port} is not posed - skipping comparison")
            continue

        if str(port) not in gold_standard_extrinsics:
            logger.warning(f"Camera {port} not found in gold standard data - skipping")
            continue

        # Parse gold standard data using the new helper function
        gold_data = gold_standard_extrinsics[str(port)]
        R_gold = parse_array_string(gold_data["rotation"])
        t_gold = parse_array_string(gold_data["translation"]).flatten()

        # Compare rotations
        rot_result = compare_rotations(cam.rotation, R_gold, port, "rotation")
        rotation_results.append(rot_result)

        # Compare translations
        trans_result = compare_translations(cam.translation, t_gold, port, "translation")
        translation_results.append(trans_result)

    # Print summary
    print("\n📊 ROTATION COMPARISON SUMMARY:")
    print("-" * 80)
    for result in rotation_results:
        print(
            f"Camera {result['port']:2d}: "
            f"Angle Error = {result['angle_error_deg']:8.4f}°, "
            f"Frobenius = {result['frobenius_error']:8.6f}, "
            f"Max Element = {result['max_element_error']:8.6f}"
        )

    print("\n📊 TRANSLATION COMPARISON SUMMARY:")
    print("-" * 80)
    for result in translation_results:
        print(
            f"Camera {result['port']:2d}: "
            f"Euclidean = {result['euclidean_error']:8.6f}, "
            f"Max Component = {result['max_component_error']:8.6f}, "
            f"Relative = {result['relative_error_percent']:6.2f}%"
        )

    # Compute aggregate statistics
    rot_angles = [r["angle_error_deg"] for r in rotation_results]
    trans_errors = [t["euclidean_error"] for t in translation_results]

    print("\n📈 AGGREGATE STATISTICS:")
    print("-" * 80)
    print(f"Rotation Angle Error: Mean = {np.mean(rot_angles):.4f}°, Max = {np.max(rot_angles):.4f}°")
    print(f"Translation Error: Mean = {np.mean(trans_errors):.6f}, Max = {np.max(trans_errors):.6f}")

    # Assertions with reasonable tolerances
    # These should FAIL initially to expose the problem
    print("\n🔍 ASSERTIONS (should initially FAIL):")
    print("-" * 80)

    # Rotation tolerance: 1 degree average error, 5 degrees max
    mean_rot_error = np.mean(rot_angles)
    max_rot_error = np.max(rot_angles)
    print(f"Mean rotation error: {mean_rot_error:.4f}° (threshold: 1.0°)")
    print(f"Max rotation error: {max_rot_error:.4f}° (threshold: 5.0°)")

    assert mean_rot_error < 1.0, f"Mean rotation error {mean_rot_error:.4f}° exceeds 1.0° threshold"
    assert max_rot_error < 5.0, f"Max rotation error {max_rot_error:.4f}° exceeds 5.0° threshold"

    # Translation tolerance: 0.05 average error, 0.1 max
    mean_trans_error = np.mean(trans_errors)
    max_trans_error = np.max(trans_errors)
    print(f"Mean translation error: {mean_trans_error:.6f} (threshold: 0.05)")
    print(f"Max translation error: {max_trans_error:.6f} (threshold: 0.1)")

    assert mean_trans_error < 0.05, f"Mean translation error {mean_trans_error:.6f} exceeds 0.05 threshold"
    assert max_trans_error < 0.1, f"Max translation error {max_trans_error:.6f} exceeds 0.1 threshold"

    print("\n✅ All assertions passed! StereoPairGraph matches gold standard.")
    print("=" * 80)


if __name__ == "__main__":
    from caliscope.logger import setup_logging

    setup_logging()
    test_stereopair_graph_against_gold_standard()
