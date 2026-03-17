"""Demo: caliscope scripting API on the 5-camera no-timestamps project.

Exercises the full calibration pipeline:
  1. Create and save a Charuco board definition
  2. Build a camera array from video metadata
  3. Calibrate intrinsics (per camera)
  4. Extract time-aligned extrinsic points
  5. Bootstrap + optimize capture volume
  6. Align to object coordinates
  7. Save and reload results

Usage:
    uv run python scripts/demo_api.py
"""

from pathlib import Path

from caliscope.api import (
    CameraArray,
    CaptureVolume,
    Charuco,
    CharucoTracker,
    calibrate_intrinsics,
    extract_image_points,
    extract_image_points_multicam,
)
from caliscope.reporting import (
    print_camera_pair_coverage,
    print_extrinsic_report,
    print_intrinsic_report,
)

# --- Project paths ---
PROJECT = Path.home() / "caliscope_projects" / "5_cam_demo_no_timestamps_clean"
INTRINSIC_DIR = PROJECT / "calibration" / "intrinsic"
EXTRINSIC_DIR = PROJECT / "calibration" / "extrinsic"
OUTPUT_DIR = PROJECT / "api_demo_output"

CAM_IDS = [0, 1, 2, 3, 4]

# --- 1. Create Charuco board and save to TOML ---
print("Step 1: Create Charuco board")
charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=5.4)
tracker = CharucoTracker(charuco)

charuco_path = OUTPUT_DIR / "charuco.toml"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
charuco.to_toml(charuco_path)
print(f"  Saved board definition to {charuco_path}")

# Verify round-trip
charuco_reloaded = Charuco.from_toml(charuco_path)
cols, rows = charuco_reloaded.columns, charuco_reloaded.rows
print(f"  Reloaded: {cols}x{rows}, square_size={charuco_reloaded.square_size_override_cm} cm")

# --- 2. Build camera array from video metadata ---
print("\nStep 2: Create CameraArray from video metadata")
intrinsic_videos = {cam_id: INTRINSIC_DIR / f"cam_{cam_id}.mp4" for cam_id in CAM_IDS}
cameras = CameraArray.from_video_metadata(intrinsic_videos)
for cam_id, cam in cameras.cameras.items():
    print(f"  cam {cam_id}: {cam.size[0]}x{cam.size[1]}")

# --- 3. Intrinsic calibration (per camera) ---
print("\nStep 3: Intrinsic calibration")
for cam_id in CAM_IDS:
    print(f"\n  Extracting intrinsic points for cam {cam_id}...")
    points = extract_image_points(
        intrinsic_videos[cam_id],
        cam_id,
        tracker,
        frame_step=10,
    )

    output = calibrate_intrinsics(points, cameras[cam_id])
    cameras[cam_id] = output.camera

    print_intrinsic_report(output)

# --- 4. Extract time-aligned extrinsic points ---
print("\nStep 4: Extract time-aligned points")
extrinsic_videos = {cam_id: EXTRINSIC_DIR / f"cam_{cam_id}.mp4" for cam_id in CAM_IDS}

ext_points = extract_image_points_multicam(extrinsic_videos, tracker, frame_step=10)
print(
    f"  {len(ext_points.df)} observations, "
    f"{ext_points.df['sync_index'].nunique()} frames, "
    f"{ext_points.df['cam_id'].nunique()} cameras"
)
print_camera_pair_coverage(ext_points)

# --- 5. Bootstrap + optimize ---
print("\nStep 5: Bootstrap and optimize")
volume = CaptureVolume.bootstrap(ext_points, cameras)

volume = volume.optimize(strict=False)
print(f"  Pass 1 RMSE: {volume.reprojection_report.overall_rmse:.3f} px")

volume = volume.filter_by_percentile_error(2.5)
volume = volume.optimize(strict=False)
print(f"  Pass 2 RMSE: {volume.reprojection_report.overall_rmse:.3f} px")

# --- 6. Align to object coordinates ---
print("\nStep 6: Align to object coordinates")
sync_idx = volume.unique_sync_indices[len(volume.unique_sync_indices) // 2]
volume = volume.align_to_object(int(sync_idx))

print_extrinsic_report(volume)

# --- 7. Save and reload ---
print("\nStep 7: Save and reload")
volume.save(OUTPUT_DIR)
print(f"  Saved capture volume to {OUTPUT_DIR}")

# Verify round-trip
volume_reloaded = CaptureVolume.load(OUTPUT_DIR)
cameras_reloaded = CameraArray.from_toml(OUTPUT_DIR / "camera_array.toml")
print(f"  Reloaded: {len(cameras_reloaded.cameras)} cameras, {len(volume_reloaded.world_points.df)} world points")

print("\nDone.")
