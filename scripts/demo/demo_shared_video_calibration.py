"""Demo: intrinsic and extrinsic calibration from the same video files.

This example uses one synchronized ChArUco recording for both calibration
stages. The recording must show varied board positions and tilts in each
camera, with enough simultaneous views to connect the camera rig.
"""

from pathlib import Path

from caliscope.api import (
    CameraArray,
    Charuco,
    CharucoTracker,
    ConstraintSet,
    calibrate_extrinsics,
    calibrate_intrinsics,
    extract_image_points_multicam,
)
from caliscope.reporting import (
    print_camera_pair_coverage,
    print_extrinsic_report,
    print_intrinsic_report,
)

# --- Recording and output paths ---
VIDEO_DIR = Path("calibration_videos")
OUTPUT_DIR = Path("capture_volume")
CAM_IDS = [0, 1, 2]

videos = {cam_id: VIDEO_DIR / f"cam_{cam_id}.mp4" for cam_id in CAM_IDS}

# Use a timestamps file when the cameras did not start at exactly the same time.
# Hardware-synchronized recordings can infer timestamps from the videos.
TIMESTAMPS: Path | None = None

# --- 1. Define the calibration target ---
# These dimensions must match the physical board in the recording.
charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=3.0)
tracker = CharucoTracker(charuco)
constraints = ConstraintSet.from_charuco(charuco)

# --- 2. Build uncalibrated cameras from the shared videos ---
cameras = CameraArray.from_video_metadata(videos)

# --- 3. Detect the board once across all synchronized videos ---
# The resulting table contains observations from every camera. Each intrinsic
# solve selects one camera's rows; the extrinsic solve uses the full table.
points = extract_image_points_multicam(
    videos,
    tracker,
    frame_step=5,
    timestamps=TIMESTAMPS,
)
print_camera_pair_coverage(points)

# --- 4. Calibrate each camera's intrinsics ---
for cam_id in CAM_IDS:
    result = calibrate_intrinsics(points, cameras[cam_id])
    cameras[cam_id] = result.camera
    print_intrinsic_report(result)

# --- 5. Calibrate the camera poses from the same observations ---
# Keep the intrinsics fixed so the two calibration stages remain distinct.
run = calibrate_extrinsics(
    points,
    cameras,
    constraints,
    refine_intrinsics=False,
)
volume = run.capture_volume
print_extrinsic_report(volume)

# --- 6. Save the calibrated camera system and observations ---
volume.save(OUTPUT_DIR)
print(f"Saved capture volume to {OUTPUT_DIR}")
