"""Demo: caliscope scripting API on the 5-camera no-timestamps project.

Exercises the full pipeline with Rich terminal reporting:
  1. Create Charuco board and tracker
  2. Extract intrinsic points (per camera, with progress bars)
  3. Calibrate intrinsics (with quality reports)
  4. Extract extrinsic points (all cameras, with progress bars)
  5. Bootstrap + optimize capture volume
  6. Print extrinsic quality report
  7. Save result to a temp directory

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
from rich.console import Console

console = Console()

# --- Project paths ---
PROJECT = Path.home() / "caliscope_projects" / "5_cam_demo_no_timestamps_clean"
INTRINSIC_DIR = PROJECT / "calibration" / "intrinsic"
EXTRINSIC_DIR = PROJECT / "calibration" / "extrinsic"
OUTPUT_DIR = PROJECT / "api_demo_output"

CAM_IDS = [0, 1, 2, 3, 4]

# --- 1. Create Charuco board (matching the project's board) ---
console.rule("[bold]Step 1: Create Charuco Board[/bold]")
charuco = Charuco(
    columns=4,
    rows=5,
    board_height=11,
    board_width=8.5,
    dictionary="DICT_4X4_50",
    units="inch",
    aruco_scale=0.75,
    square_size_override_cm=5.4,
    inverted=False,
    legacy_pattern=False,
)
tracker = CharucoTracker(charuco)
console.print(f"  Board: {charuco.columns}x{charuco.rows}, square_size={charuco.square_size_override_cm} cm")
console.print()

# --- 2. Build camera array from video metadata ---
console.rule("[bold]Step 2: Create CameraArray from Video Metadata[/bold]")
intrinsic_videos = {cam_id: INTRINSIC_DIR / f"cam_{cam_id}.mp4" for cam_id in CAM_IDS}
cameras = CameraArray.from_video_metadata(intrinsic_videos)
for cam_id, cam in cameras.cameras.items():
    console.print(f"  cam {cam_id}: {cam.size[0]}x{cam.size[1]}")
console.print()

# --- 3. Intrinsic calibration (per camera) ---
console.rule("[bold]Step 3: Intrinsic Calibration[/bold]")
for cam_id in CAM_IDS:
    console.print(f"\n[bold cyan]Extracting intrinsic points for cam {cam_id}...[/bold cyan]")
    points = extract_image_points(
        intrinsic_videos[cam_id],
        cam_id,
        tracker,
        frame_step=10,
    )
    console.print(f"  Detected {len(points.df)} observations across {points.df['sync_index'].nunique()} frames")

    output = calibrate_intrinsics(points, cameras[cam_id])
    cameras[cam_id] = output.camera

    print_intrinsic_report(output, console=console)

# --- 4. Extrinsic calibration ---
console.rule("[bold]Step 4: Extract Time-Aligned Points[/bold]")
extrinsic_videos = {cam_id: EXTRINSIC_DIR / f"cam_{cam_id}.mp4" for cam_id in CAM_IDS}

ext_points = extract_image_points_multicam(extrinsic_videos, tracker, frame_step=10)
console.print(
    f"\n  Total: {len(ext_points.df)} observations, "
    f"{ext_points.df['sync_index'].nunique()} frames, "
    f"{ext_points.df['cam_id'].nunique()} cameras"
)
print_camera_pair_coverage(ext_points, console=console)

# --- 5. Bootstrap + optimize ---
console.rule("[bold]Step 5: Bootstrap & Optimize[/bold]")
console.print("  Bootstrapping extrinsic poses...")
volume = CaptureVolume.bootstrap(ext_points, cameras)
console.print("  Bootstrap complete. Optimizing (pass 1)...")

volume = volume.optimize(strict=False)
console.print(f"  Pass 1 done. RMSE: {volume.reprojection_report.overall_rmse:.3f} px")

console.print("  Filtering outliers (2.5th percentile)...")
volume = volume.filter_by_percentile_error(2.5)

console.print("  Optimizing (pass 2)...")
volume = volume.optimize(strict=False)
console.print(f"  Pass 2 done. RMSE: {volume.reprojection_report.overall_rmse:.3f} px")

# --- 6. Align to object (pick a frame with good coverage) ---
console.rule("[bold]Step 6: Align to Object Coordinates[/bold]")
sync_idx = volume.unique_sync_indices[len(volume.unique_sync_indices) // 2]
console.print(f"  Aligning to object coordinates at sync_index={sync_idx}...")
volume = volume.align_to_object(int(sync_idx))
console.print("  Alignment complete.")

# --- 7. Print extrinsic report ---
console.rule("[bold]Step 7: Extrinsic Quality Report[/bold]")
print_extrinsic_report(volume, console=console)

# --- 8. Save ---
console.rule("[bold]Step 8: Save[/bold]")
volume.save(OUTPUT_DIR)
console.print(f"  Saved to {OUTPUT_DIR}")
console.print()
console.rule("[bold green]Demo complete![/bold green]")
