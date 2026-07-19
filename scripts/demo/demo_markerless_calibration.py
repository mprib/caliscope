"""Markerless multicamera calibration on Pose2Sim data, via the caliscope public API.

RTMPose Halpe26 keypoints from raw video, epipolar bootstrap and bundle
adjustment with no calibration object, GeoCalib vertical estimation, a single
tape-measured camera baseline for metric scale, and Blender scene export.

Inputs:
    - Pose2Sim Demo_SinglePerson videos (4 cameras, 100 frames, 60 fps)
    - Qualisys QCA intrinsics (measured, converted to pixels)
    - One camera-pair distance (cam1-cam2, tape-measured from QCA)

Outputs:
    - Calibrated capture volume (camera_array.toml, image/world points)
    - Blender scene with animated skeleton and camera backgrounds
    - Inter-camera distance comparison against the full Qualisys reference

Prerequisites:
    - Pose2Sim Demo_SinglePerson data in DATA_DIR (see Reproduce section below)
    - RTMPose-l Halpe26 weights downloaded via the caliscope GUI or CLI

Reproduce the data:
    git clone --no-checkout --depth 1 --filter=blob:none \
        https://github.com/perfanalytics/pose2sim.git p2s
    cd p2s && git sparse-checkout set --no-cone Pose2Sim/Demo_SinglePerson \
        && git checkout
    cp -r Pose2Sim/Demo_SinglePerson/* <DATA_DIR>

Run:
    uv run python scripts/demo/demo_markerless_calibration.py
"""

from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np

from caliscope import MODELS_DIR
from caliscope.api import (
    CameraArray,
    CameraData,
    CameraDistance,
    calibrate_extrinsics,
    estimate_vertical,
    extract_image_points_multicam,
    write_blender_scene,
)
from caliscope.trackers import tracker_registry

DATA_DIR = Path("~/projects/pose2sim/Pose2Sim/Demo_SinglePerson").expanduser()
OUTPUT_DIR = DATA_DIR / "demo_markerless" / "rtmpose"
TRACKER_KEY = "ONNX_rtmpose_l_halpe26"

# ── INPUTS ───────────────────────────────────────────────────────────────────

# Qualisys QCA intrinsics converted to pixels. Focal and cx rescaled from the
# 1088 px sensor to actual video width for cam 1 and 2 (1080 px). Distortion
# reordered from QCA (radial1, radial2, radial3, tangental1, tangental2) to
# OpenCV (k1, k2, p1, p2, k3).
qca_distortion = {
    1: np.array([-0.046183, 0.139983, 0.000608, 0.00069, 0.0]),
    2: np.array([-0.047847, 0.136786, 0.000972, 0.000291, 0.0]),
    3: np.array([-0.046705, 0.137622, -0.000542, -0.000517, 0.0]),
    4: np.array([-0.047633, 0.134667, 0.000277, 0.000199, 0.0]),
}
cameras = CameraArray(
    cameras={
        1: CameraData.from_intrinsics(
            cam_id=1,
            size=(1080, 1920),
            focal_length=1668.9,
            cx=529.1,
            cy=948.1,
            distortions=qca_distortion[1],
        ),
        2: CameraData.from_intrinsics(
            cam_id=2,
            size=(1080, 1920),
            focal_length=1661.4,
            cx=530.6,
            cy=963.2,
            distortions=qca_distortion[2],
        ),
        3: CameraData.from_intrinsics(
            cam_id=3,
            size=(1088, 1920),
            focal_length=1681.6,
            cx=513.2,
            cy=955.0,
            distortions=qca_distortion[3],
        ),
        4: CameraData.from_intrinsics(
            cam_id=4,
            size=(1088, 1920),
            focal_length=1675.2,
            cx=540.1,
            cy=964.0,
            distortions=qca_distortion[4],
        ),
    }
)

videos = {cam_id: DATA_DIR / "videos" / f"cam{cam_id:02d}.mp4" for cam_id in cameras.cameras}

# One tape-measured baseline for metric scale. Only this pair enters the
# calibration; the remaining five pairwise distances are scored against
# Qualisys as a validation.
SCALE_CUE = CameraDistance(cam_a=1, cam_b=2, meters=2.85354)

# Qualisys reference camera centers in metres. Validation only.
qca_centers_m = {
    1: np.array([1.46023, -1.90916, 1.89651]),
    2: np.array([2.58201, 0.70657, 1.69098]),
    3: np.array([-3.21687, 2.23118, 2.08819]),
    4: np.array([-3.75872, -1.41567, 1.88179]),
}

t_total = perf_counter()
timings: dict[str, float] = {}

# ── 1. KEYPOINTS ─────────────────────────────────────────────────────────────

print("\nExtracting keypoints...")
tracker_registry.scan_onnx_models(MODELS_DIR)
tracker = tracker_registry.create(TRACKER_KEY)
t0 = perf_counter()
image_points = extract_image_points_multicam(videos, tracker, frame_step=1)
timings["extraction"] = perf_counter() - t0
print(f"  {len(image_points.df)} observations in {timings['extraction']:.1f}s")

# ── 2. EXTRINSICS ────────────────────────────────────────────────────────────

print("\nCalibrating extrinsics...")
t0 = perf_counter()
run = calibrate_extrinsics(image_points, cameras, constraints=None, refine_intrinsics=False)
timings["extrinsics"] = perf_counter() - t0
volume = run.capture_volume
print(f"  reprojection RMSE {volume.reprojection_report.overall_rmse:.2f} px in {timings['extrinsics']:.1f}s")

# ── 3. ANCHORING ─────────────────────────────────────────────────────────────

print("\nEstimating vertical (GeoCalib)...")
t0 = perf_counter()
vertical = estimate_vertical(videos, cameras, frames_per_camera=1)
timings["vertical"] = perf_counter() - t0
print(f"  done in {timings['vertical']:.1f}s")

print("\nAnchoring (orient, scale, ground, center)...")
t0 = perf_counter()
volume = volume.oriented(up=vertical.up_per_cam)
volume = volume.scaled(SCALE_CUE)
volume = volume.grounded()
volume = volume.centered()
timings["anchoring"] = perf_counter() - t0
print(f"  done in {timings['anchoring']:.2f}s")

# ── 4. COMPARISON VS QUALISYS ────────────────────────────────────────────────

print("\nComparing against Qualisys reference...")
markerless_centers = {
    cam_id: -cam.rotation.T @ cam.translation for cam_id, cam in volume.camera_array.posed_cameras.items()
}
errors_mm = []
for cam_a, cam_b in combinations(sorted(markerless_centers), 2):
    markerless_m = float(np.linalg.norm(markerless_centers[cam_a] - markerless_centers[cam_b]))
    reference_m = float(np.linalg.norm(qca_centers_m[cam_a] - qca_centers_m[cam_b]))
    err_mm = (markerless_m - reference_m) * 1000
    err_pct = 100 * (markerless_m - reference_m) / reference_m
    is_ref = cam_a == SCALE_CUE.cam_a and cam_b == SCALE_CUE.cam_b
    tag = " (scale reference)" if is_ref else ""
    errors_mm.append(err_mm)
    print(
        f"  cam{cam_a}-cam{cam_b}: markerless {markerless_m * 1000:7.1f} mm, "
        f"QCA {reference_m * 1000:7.1f} mm, error {err_mm:+6.1f} mm ({err_pct:+.2f}%){tag}"
    )
rmse_mm = float(np.sqrt(np.mean(np.square(errors_mm))))
print(f"  RMSE vs Qualisys: {rmse_mm:.1f} mm over {len(errors_mm)} pairs")

# ── 5. BLENDER EXPORT ────────────────────────────────────────────────────────

print("\nTriangulating world points...")
t0 = perf_counter()
world_points = image_points.triangulate(volume.camera_array)
timings["triangulation"] = perf_counter() - t0
print(f"  {len(world_points.df)} points in {timings['triangulation']:.1f}s")

print("\nExporting Blender scene...")
t0 = perf_counter()
volume.save(OUTPUT_DIR)
scene_script = write_blender_scene(
    volume.camera_array,
    world_points,
    OUTPUT_DIR / "capture_volume_scene.py",
    videos=videos,
    wireframe=tracker_registry.wireframe_for(TRACKER_KEY),
)
timings["blender"] = perf_counter() - t0
print(f"  {scene_script.with_suffix('.blend')} in {timings['blender']:.1f}s")

# ── SUMMARY ──────────────────────────────────────────────────────────────────

timings["total"] = perf_counter() - t_total
print()
print("\nTiming summary:")
for stage, seconds in timings.items():
    print(f"  {stage:<15} {seconds:6.1f}s")

blend_path = scene_script.with_suffix(".blend")
print()
print(f"\nOpen the scene:\n  blender {blend_path}")
