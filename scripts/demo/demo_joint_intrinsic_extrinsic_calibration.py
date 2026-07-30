"""Experimental demo: recover intrinsics and extrinsics in one calibration.

The cameras begin with image dimensions only. Caliscope supplies an initial
pinhole estimate, then refines focal length and radial distortion together
with camera poses and 3D points.
"""

from pathlib import Path

from caliscope.api import (
    CameraArray,
    Charuco,
    CharucoTracker,
    ConstraintSet,
    calibrate_extrinsics,
    extract_image_points_multicam,
)
from caliscope.reporting import print_extrinsic_report

# --- Recording and output paths ---
VIDEO_DIR = Path("calibration_videos")
OUTPUT_DIR = Path("capture_volume")
CAM_IDS = [0, 1, 2]

videos = {cam_id: VIDEO_DIR / f"cam_{cam_id}.mp4" for cam_id in CAM_IDS}
TIMESTAMPS: Path | None = None

# --- 1. Define the calibration target ---
# These dimensions must match the physical board in the recording.
charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=3.0)
tracker = CharucoTracker(charuco)
constraints = ConstraintSet.from_charuco(charuco)

# --- 2. Build cameras with image dimensions but no calibration ---
cameras = CameraArray.from_video_metadata(videos)

# --- 3. Detect the board across the synchronized videos ---
points = extract_image_points_multicam(
    videos,
    tracker,
    frame_step=5,
    timestamps=TIMESTAMPS,
)

# --- 4. Recover intrinsics and extrinsics together ---
# Each standard camera starts with fx = fy = image_width / 2, a centered
# principal point, and zero distortion. Bundle adjustment refines focal length,
# k1, and k2. Move the board toward and away from every camera so those
# parameters can be distinguished from camera position and scene scale.
run = calibrate_extrinsics(
    points,
    cameras,
    constraints,
    refine_intrinsics=True,
)
volume = run.capture_volume

if run.intrinsic_refinement_gated:
    print("Intrinsic refinement was disabled because the recording lacked depth variation.")

print_extrinsic_report(volume)
volume.save(OUTPUT_DIR)
