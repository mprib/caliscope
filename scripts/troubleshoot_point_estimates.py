import rtoml
import numpy as np
import pandas as pd
from caliscope import __root__

# Paths
GOLD_PATH = __root__ / "tests/reference/stereograph_gold_standard/gold_point_estimates.toml"
NEW_PATH = __root__ / "tests/reference/stereograph_gold_standard/new_point_estimates.toml"
# We need the camera array to map indices back to ports for a true comparison
# Assuming standard port-to-index mapping (0->0, 1->1) if unavailable,
# but ideally we load the matching camera_array.json.
# For this script, we will assume camera_index is consistent between runs.


def load_data(path, label):
    with open(path, "r") as f:
        data = rtoml.load(f)

    # Extract Arrays
    sync = np.array(data["sync_indices"])
    pid = np.array(data["point_id"])
    cam = np.array(data["camera_indices"])
    img = np.array(data["img"])
    obj_indices = np.array(data["obj_indices"])
    obj = np.array(data["obj"])

    # Create 2D DataFrame (Observations)
    df_2d = pd.DataFrame(
        {
            "sync_index": sync,
            "point_id": pid,
            "camera_index": cam,
            f"x_2d_{label}": img[:, 0],
            f"y_2d_{label}": img[:, 1],
            "obj_idx_ptr": obj_indices,
        }
    )

    # Create 3D DataFrame (Unique Points)
    # We map the 3D coordinates onto the 2D observations first to handle the indirection
    df_2d[f"x_3d_{label}"] = obj[obj_indices, 0]
    df_2d[f"y_3d_{label}"] = obj[obj_indices, 1]
    df_2d[f"z_3d_{label}"] = obj[obj_indices, 2]

    return df_2d


def compare_robust():
    print("Loading Data...")
    df_gold = load_data(GOLD_PATH, "gold")
    df_new = load_data(NEW_PATH, "new")

    print(f"  Gold Rows: {len(df_gold)}")
    print(f"  New Rows:  {len(df_new)}")

    # MERGE on Sync + PointID + Camera to ensure we compare apples to apples
    merged = pd.merge(df_gold, df_new, on=["sync_index", "point_id", "camera_index"], how="inner")

    print(f"  Matched Rows: {len(merged)}")

    # 1. Verify 2D Inputs (Observations)
    diff_2d_x = merged["x_2d_new"] - merged["x_2d_gold"]
    diff_2d_y = merged["y_2d_new"] - merged["y_2d_gold"]
    dist_2d = np.sqrt(diff_2d_x**2 + diff_2d_y**2)

    print("\n--- 2D Input Data Comparison ---")
    if dist_2d.max() < 0.001:
        print("  ✅ 2D Inputs are IDENTICAL. (Sorting issue resolved)")
    else:
        print(f"  ❌ 2D Inputs DIFFER! Max Diff: {dist_2d.max():.4f} pixels")
        print("     This implies one branch is using Raw points and the other Undistorted.")

    # 2. Verify 3D Estimates (Triangulation)
    diff_3d_x = merged["x_3d_new"] - merged["x_3d_gold"]
    diff_3d_y = merged["y_3d_new"] - merged["y_3d_gold"]
    diff_3d_z = merged["z_3d_new"] - merged["z_3d_gold"]
    dist_3d = np.sqrt(diff_3d_x**2 + diff_3d_y**2 + diff_3d_z**2)

    print("\n--- 3D Estimate Comparison ---")
    print(f"  Mean Shift:   {dist_3d.mean():.4f}")
    print(f"  Max Shift:    {dist_3d.max():.4f}")

    # Check for systematic offset
    mean_diff = np.array([diff_3d_x.mean(), diff_3d_y.mean(), diff_3d_z.mean()])
    print(f"  Systematic Offset (XYZ): {mean_diff}")


if __name__ == "__main__":
    compare_robust()
