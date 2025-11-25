import json
import numpy as np
import sys
from caliscope import __root__

# Paths
GOLD_PATH = __root__ / "tests/reference/stereograph_gold_standard/main_initial_camera_array.json"
NEW_PATH = __root__ / "tests/reference/stereograph_gold_standard/new_initial_camera_array.json"


def parse_vector(v_str):
    """Parses the string representation of numpy arrays stored in the JSON"""
    if not isinstance(v_str, str):
        return np.array(v_str)
    # Remove brackets and clean whitespace
    clean = v_str.replace("[", "").replace("]", "").replace("\n", " ").strip()
    # Handle comma or space separation
    sep = "," if "," in clean else " "
    return np.fromstring(clean, sep=sep)


def check_cameras():
    print(f"Loading Gold: {GOLD_PATH}")
    with open(GOLD_PATH, "r") as f:
        gold = json.load(f)

    print(f"Loading New:  {NEW_PATH}")
    if not NEW_PATH.exists():
        print("❌ CRITICAL: 'new_initial_camera_array.json' not found.")
        print("   Did test_calibration.py run far enough to save it?")
        sys.exit(1)

    with open(NEW_PATH, "r") as f:
        new = json.load(f)

    print("\n--- Camera Extrinsics Comparison ---")

    ports = sorted([k for k in new.keys()])

    for port in ports:
        if port not in gold:
            print(f"⚠️  Camera {port}: Present in NEW but missing in GOLD.")
            continue

        # Extract
        g_rot = parse_vector(gold[port]["rotation"])
        n_rot = parse_vector(new[port]["rotation"])
        g_trans = parse_vector(gold[port]["translation"]).flatten()
        n_trans = parse_vector(new[port]["translation"]).flatten()

        # 1. Check for NaNs (The #1 cause of optimization hangs)
        if np.isnan(n_rot).any() or np.isnan(n_trans).any():
            print(f"🚨 Camera {port} [CRITICAL]: Contains NaNs!")
            print(f"   Rot: {n_rot}")
            print(f"   Trans: {n_trans}")
            print("   >>> THIS IS CAUSING THE HANG <<<")
            continue

        # 2. Check for Extremes (e.g. Translation > 100 meters)
        if np.max(np.abs(n_trans)) > 100:
            print(f"⚠️  Camera {port} [SUSPICIOUS]: Large translation detected.")
            print(f"   New Trans: {n_trans}")

        # 3. Compare with Gold
        rot_dist = np.linalg.norm(g_rot - n_rot)
        trans_dist = np.linalg.norm(g_trans - n_trans)

        print(f"Camera {port}:")
        print(f"  Rotation Diff:    {rot_dist:.4f}")
        print(f"  Translation Diff: {trans_dist:.4f}")

        if trans_dist > 0.1:
            print(f"  -> Gold T: {g_trans}")
            print(f"  -> New  T: {n_trans}")


if __name__ == "__main__":
    check_cameras()
