"""Markerless multicamera calibration on OpenCap LabValidation data, via the caliscope public API.

RTMPose Halpe26 keypoints from synchronized video, epipolar bootstrap and
bundle adjustment with no calibration object, GeoCalib vertical estimation, a
single camera baseline (the largest, read from the OpenCap pickle) for metric
scale, and Blender scene export.

The five iPhone cameras share known intrinsics and known extrinsics from
OpenCap's own checkerboard calibration. This script ignores those extrinsics for
the solve and recovers them from body motion alone, then scores the markerless
result against the checkerboard reference.

OpenCap's cameras are not hardware-synced; their "*_syncdWithMocap.avi" videos
are trimmed to the mocap timeline, not aligned camera-to-camera, and carry a
constant per-camera frame offset up to ~6 frames. The alignment stage below
measures and removes it. The default trial is STS1 (sit-to-stand): the subject
stays centered and fully framed in all five cameras, with clear vertical motion
for the sync cross-correlation to lock, slow enough that the residual sub-frame
offset is invisible. A drop jump (--trial DJ1) exposes the sub-frame floor;
overground walking cannot be synced here because the subject walks toward the
center camera, which sees looming, not the vertical bob the sync relies on.

Inputs:
    - OpenCap subject2 Session0 STS1 videos (5 cameras, ~510 frames, 60 fps, 720x1280)
    - Per-camera intrinsics from each Cam{N}/cameraIntrinsicsExtrinsics.pickle
    - One camera-pair distance (largest baseline), derived from the pickle extrinsics

Outputs:
    - Calibrated capture volume (camera_array.toml, image/world points)
    - Blender scene with animated skeleton and camera backgrounds
    - All-intra H.264 backgrounds under <output-dir>/videos, co-located with the .blend (self-contained snapshot)
    - Procrustes and inter-camera distance comparison against the pickle reference

Prerequisites:
    - OpenCap LabValidation subject2 data (see Reproduce section below)
    - RTMPose-l Halpe26 and GeoCalib weights downloaded via the caliscope GUI or CLI

Reproduce the data:
    OpenCap LabValidation is published on SimTK (Apache 2.0). From the project
    page https://simtk.org/projects/opencap download the LabValidation dataset,
    then point --data-dir at the subject2 folder. It must contain:
        VideoData/Session0/Cam{0..4}/cameraIntrinsicsExtrinsics.pickle
        VideoData/Session0/Cam{0..4}/STS1/STS1_syncdWithMocap.avi
    Data credit: Uhlrich et al., 2023, PLOS Computational Biology. Subject6 opted
    out and is excluded; other participants consented to identifiable video release.

Run:
    uv run python scripts/demo/demo_opencap_calibration.py \
        --data-dir /path/to/opencap/LabValidation/subject2 \
        --output-dir demo_output
"""

import argparse
import pickle
from itertools import combinations
from pathlib import Path
from time import perf_counter

import av
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
from caliscope.core.point_data import ImagePoints
from caliscope.trackers import tracker_registry

TRACKER_KEY = "ONNX_rtmpose_l_halpe26"
CAM_IDS = list(range(5))  # OpenCap Cam0..Cam4
REF_CAM = 0  # temporal-alignment reference (OpenCap convention: first camera)

# The pickle imageSize is stale landscape [1280, 720]; the intrinsic matrix
# (cx~360, cy~640) and the synced videos are portrait 720x1280 (width, height).
IMAGE_SIZE = (720, 1280)


def _cam_dir(data_dir: Path, cam_id: int, session: str = "Session0") -> Path:
    return data_dir / "VideoData" / session / f"Cam{cam_id}"


def _load_pickles(data_dir: Path, session: str = "Session0") -> dict[int, dict]:
    pickles = {}
    for cam_id in CAM_IDS:
        path = _cam_dir(data_dir, cam_id, session) / "cameraIntrinsicsExtrinsics.pickle"
        with open(path, "rb") as f:
            pickles[cam_id] = pickle.load(f)
    return pickles


def _build_cameras(pickles: dict[int, dict]) -> CameraArray:
    """One CameraData per pickle, fx and fy kept distinct (they differ slightly)."""
    cameras = {}
    for cam_id, d in pickles.items():
        K = np.asarray(d["intrinsicMat"], dtype=np.float64)
        cameras[cam_id] = CameraData.from_intrinsics(
            cam_id=cam_id,
            size=IMAGE_SIZE,
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            distortions=np.asarray(d["distortion"], dtype=np.float64).ravel(),
        )
    return CameraArray(cameras=cameras)


def _pickle_center_mm(d: dict) -> np.ndarray:
    """Camera center in millimeters from the pickle extrinsics: C = -R.T @ t."""
    R = np.asarray(d["rotation"], dtype=np.float64)
    t = np.asarray(d["translation"], dtype=np.float64).reshape(3)
    return -R.T @ t


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid rotation and translation mapping P onto Q (no scaling)."""
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cQ - R @ cP


def _transcode_all_intra(
    src: Path, dst: Path, *, fps: int = 60, crf: int = 18, skip: int = 0, count: int | None = None
) -> int:
    """Re-encode a video to all-intra H.264 (gop size 1), returning frames written.

    Blender scrubs non-all-intra codecs by snapping to the nearest I-frame, so
    inter-coded backgrounds drift out of sync between cameras at the same scene
    frame. Making every frame a keyframe keeps each camera frame-accurate under
    scrubbing. Re-encoding on a fresh 60 fps timeline also drops the source AVI's
    phantom timeline slots (its header counts 170 for 166 real frames).

    `skip` drops that many leading source frames and `count` caps the output
    length, so the emitted file's frame index equals the aligned sync index. This
    is what keeps the Blender backgrounds honest with the calibrated 3D after the
    temporal-alignment stage shifts each camera onto a common timeline.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with av.open(str(src)) as ic, av.open(str(dst), "w") as oc:
        istream = ic.streams.video[0]
        ostream = oc.add_stream("libx264", rate=fps)
        ostream.width = istream.codec_context.width
        ostream.height = istream.codec_context.height
        ostream.pix_fmt = "yuv420p"
        ostream.codec_context.gop_size = 1
        ostream.options = {"crf": str(crf), "x264-params": "keyint=1:min-keyint=1:scenecut=0"}
        for i, frame in enumerate(ic.decode(istream)):
            if i < skip or (count is not None and written >= count):
                continue
            frame.pts = None
            for packet in ostream.encode(frame):
                oc.mux(packet)
            written += 1
        for packet in ostream.encode():
            oc.mux(packet)
    return written


# ─── SPIKE: video-only temporal alignment ────────────────────────────────────
# OpenCap's "syncdWithMocap" videos are trimmed to the mocap window, not aligned
# camera-to-camera; a constant per-camera integer frame offset of up to ~6 frames
# remains. Left uncorrected it inflates bundle-adjustment reprojection error and
# desynchronizes the Blender backgrounds. This measures the offset from the
# tracked keypoints and shifts each camera onto a common timeline. Spike for the
# real feature (ImagePoints.temporally_aligned); see specs/video-sync-research.md.


def _camera_motion_signal(df, cam_id: int, sync_range: range) -> np.ndarray:
    """Per-frame vertical motion signal for one camera: mean keypoint y, gap-filled."""
    sub = df[df["cam_id"] == cam_id]
    y = sub.groupby("sync_index")["img_loc_y"].mean()
    return y.reindex(sync_range).interpolate().bfill().ffill().to_numpy()


def _best_lag(ref: np.ndarray, sig: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Integer lag maximizing normalized cross-correlation. Positive: sig leads ref."""
    r0 = (ref - ref.mean()) / ref.std()
    s0 = (sig - sig.mean()) / sig.std()
    best, best_r = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = float(np.corrcoef(r0[lag:], s0[: len(s0) - lag])[0, 1])
        else:
            r = float(np.corrcoef(r0[:lag], s0[-lag:])[0, 1])
        if r > best_r:
            best_r, best = r, lag
    return best, best_r


def _measure_lags(image_points, ref_cam: int, max_lag: int = 10) -> dict[int, int]:
    """Per-camera integer frame lag vs the reference, from keypoint cross-correlation.

    Correlation alone is ambiguous on periodic motion (thin peak margin); the
    reprojection refinement below settles the exact integer. See the research note.
    """
    df = image_points.df
    lo, hi = int(df["sync_index"].min()), int(df["sync_index"].max())
    rng = range(lo, hi + 1)
    ref_sig = _camera_motion_signal(df, ref_cam, rng)
    lags = {}
    for cam_id in sorted(int(c) for c in df["cam_id"].unique()):
        lag, r = _best_lag(ref_sig, _camera_motion_signal(df, cam_id, rng), max_lag)
        lags[cam_id] = lag
        print(f"    cam{cam_id}: lag {lag:+d} frames (r={r:.4f})")
    return lags


def _apply_lags(image_points, lags: dict[int, int]):
    """Shift each camera's sync_index by its lag, trim to the common window, rebase to 0."""
    df = image_points.df
    df["sync_index"] = df["sync_index"] + df["cam_id"].map(lags)
    lo = int(df.groupby("cam_id")["sync_index"].min().max())
    hi = int(df.groupby("cam_id")["sync_index"].max().min())
    df = df[(df["sync_index"] >= lo) & (df["sync_index"] <= hi)].copy()
    df["sync_index"] = df["sync_index"] - lo
    return ImagePoints(df), lo, hi


def _refine_lags_by_reprojection(image_points, cameras, lags: dict[int, int], ref_cam: int) -> dict[int, int]:
    """Coordinate-descent on integer lags, minimizing bundle-adjustment RMSE.

    The correlation seed is ambiguous under periodic gait; reprojection error is
    the honest arbiter (OpenCap does the same via triangulation). One pass over
    the non-reference cameras, +/-2 frames around the seed, keeping any RMSE gain.
    """

    def rmse(candidate: dict[int, int]) -> float:
        aligned, _, _ = _apply_lags(image_points, candidate)
        run = calibrate_extrinsics(aligned, cameras, constraints=None, refine_intrinsics=False)
        return float(run.capture_volume.reprojection_report.overall_rmse)

    best, best_rmse = dict(lags), rmse(lags)
    print(f"    seed RMSE {best_rmse:.2f} px")
    for cam_id in sorted(lags):
        if cam_id == ref_cam:
            continue
        for delta in (-2, -1, 1, 2):
            trial = dict(best)
            trial[cam_id] = lags[cam_id] + delta
            r = rmse(trial)
            if r < best_rmse:
                best_rmse, best = r, trial
    print(f"    refined RMSE {best_rmse:.2f} px, lags { {c: best[c] for c in sorted(best)} }")
    return best


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Markerless multicamera calibration on OpenCap LabValidation data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="OpenCap LabValidation subject2 directory (contains VideoData/Session0/Cam{0..4}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_output"),
        help="Where to write the calibrated snapshot and Blender scene (default: demo_output).",
    )
    parser.add_argument(
        "--session",
        default="Session0",
        help="Session folder under VideoData/ (default: Session0). Walking trials live in Session1, "
        "which has its own per-session calibration pickle.",
    )
    parser.add_argument(
        "--trial",
        default="STS1",
        help=(
            "Trial name under each Cam{N}/ (default: STS1 sit-to-stand). A slow, centered trial "
            "(STS1, squats1) hides the sub-frame sync residual that a drop jump (DJ1) exposes; "
            "keypoint cross-correlation needs motion (static1 is too still) and consistent framing "
            "in all cameras (overground walking transits out of view and cannot sync)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if not data_dir.exists():
        raise SystemExit(f"--data-dir not found: {data_dir}")

    t_total = perf_counter()
    timings: dict[str, float] = {}

    # ── INPUTS ───────────────────────────────────────────────────────────────

    session = args.session
    pickles = _load_pickles(data_dir, session)
    cameras = _build_cameras(pickles)
    trial = args.trial
    videos = {cam_id: _cam_dir(data_dir, cam_id, session) / trial / f"{trial}_syncdWithMocap.avi" for cam_id in CAM_IDS}
    print(f"Session: {session}  Trial: {trial}")

    # Largest camera baseline from the pickle extrinsics, the most stable scale cue.
    pickle_centers_mm = {cam_id: _pickle_center_mm(pickles[cam_id]) for cam_id in CAM_IDS}
    scale_a, scale_b = max(
        combinations(CAM_IDS, 2),
        key=lambda pair: np.linalg.norm(pickle_centers_mm[pair[0]] - pickle_centers_mm[pair[1]]),
    )
    scale_m = float(np.linalg.norm(pickle_centers_mm[scale_a] - pickle_centers_mm[scale_b]) / 1000.0)
    scale_cue = CameraDistance(cam_a=scale_a, cam_b=scale_b, meters=scale_m)
    print(f"Scale cue: cam{scale_a}-cam{scale_b} baseline {scale_m:.4f} m (largest, from pickle extrinsics)")

    # ── 1. KEYPOINTS ─────────────────────────────────────────────────────────

    print("\nExtracting keypoints...")
    tracker_registry.scan_onnx_models(MODELS_DIR)
    tracker = tracker_registry.create(TRACKER_KEY)
    t0 = perf_counter()
    image_points = extract_image_points_multicam(videos, tracker, frame_step=1)
    timings["extraction"] = perf_counter() - t0
    print(f"  {len(image_points.df)} observations in {timings['extraction']:.1f}s")

    # ── 1b. TEMPORAL ALIGNMENT ───────────────────────────────────────────────
    # OpenCap's "syncdWithMocap" videos carry a constant per-camera frame offset.
    # Measure it from the keypoints, settle the integer by reprojection, and shift
    # every camera onto a common timeline before calibrating.

    print("\nMeasuring per-camera frame offsets...")
    t0 = perf_counter()
    seed_lags = _measure_lags(image_points, ref_cam=REF_CAM)
    lags = _refine_lags_by_reprojection(image_points, cameras, seed_lags, ref_cam=REF_CAM)
    image_points, window_lo, window_hi = _apply_lags(image_points, lags)
    n_aligned = window_hi - window_lo + 1
    timings["alignment"] = perf_counter() - t0
    print(f"  aligned to {n_aligned} common frames in {timings['alignment']:.1f}s")

    # ── 2. EXTRINSICS ────────────────────────────────────────────────────────

    print("\nCalibrating extrinsics...")
    t0 = perf_counter()
    run = calibrate_extrinsics(image_points, cameras, constraints=None, refine_intrinsics=False)
    timings["extrinsics"] = perf_counter() - t0
    volume = run.capture_volume
    print(f"  reprojection RMSE {volume.reprojection_report.overall_rmse:.2f} px in {timings['extrinsics']:.1f}s")

    # ── 3. ANCHORING ─────────────────────────────────────────────────────────

    print("\nEstimating vertical (GeoCalib)...")
    t0 = perf_counter()
    vertical = estimate_vertical(videos, cameras, frames_per_camera=1)
    timings["vertical"] = perf_counter() - t0
    print(f"  done in {timings['vertical']:.1f}s")

    print("\nAnchoring (orient, scale, ground, center)...")
    t0 = perf_counter()
    volume = volume.oriented(up=vertical.up_per_cam)
    volume = volume.scaled(scale_cue)
    volume = volume.grounded()
    volume = volume.centered()
    timings["anchoring"] = perf_counter() - t0
    print(f"  done in {timings['anchoring']:.2f}s")

    # ── 4. VALIDATION VS OPENCAP PICKLE ──────────────────────────────────────

    print("\nComparing against the OpenCap pickle calibration...")
    cam_ids = sorted(volume.camera_array.posed_cameras)
    cali_centers_mm = {
        cam_id: (-cam.rotation.T @ cam.translation) * 1000.0
        for cam_id, cam in volume.camera_array.posed_cameras.items()
    }

    P = np.array([cali_centers_mm[c] for c in cam_ids])
    Q = np.array([pickle_centers_mm[c] for c in cam_ids])
    R_align, T_align = _kabsch(P, Q)
    P_aligned = (R_align @ P.T).T + T_align

    print("  per-camera pose error after rigid Procrustes (no scaling):")
    for i, c in enumerate(cam_ids):
        center_err_mm = float(np.linalg.norm(P_aligned[i] - Q[i]))
        R_cali = np.asarray(volume.camera_array.cameras[c].rotation, dtype=np.float64)
        R_ref = np.asarray(pickles[c]["rotation"], dtype=np.float64)
        R_err = R_ref @ (R_cali @ R_align.T).T
        rot_err_deg = float(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))))
        print(f"    cam{c}: center {center_err_mm:6.1f} mm, rotation {rot_err_deg:5.2f} deg")

    print("  inter-camera distance (caliscope vs pickle):")
    errors_mm = []
    for cam_a, cam_b in combinations(cam_ids, 2):
        cali_mm = float(np.linalg.norm(cali_centers_mm[cam_a] - cali_centers_mm[cam_b]))
        ref_mm = float(np.linalg.norm(pickle_centers_mm[cam_a] - pickle_centers_mm[cam_b]))
        err_mm = cali_mm - ref_mm
        err_pct = 100.0 * err_mm / ref_mm
        tag = " (scale reference)" if (cam_a, cam_b) == (scale_a, scale_b) else ""
        errors_mm.append(err_mm)
        print(
            f"    cam{cam_a}-cam{cam_b}: caliscope {cali_mm:7.1f} mm, "
            f"pickle {ref_mm:7.1f} mm, error {err_mm:+6.1f} mm ({err_pct:+.2f}%){tag}"
        )
    rmse_mm = float(np.sqrt(np.mean(np.square(errors_mm))))
    print(f"  RMSE vs pickle: {rmse_mm:.1f} mm over {len(errors_mm)} pairs")
    print("  OpenCap checkerboard benchmark: sub-cm, sub-0.25-degree.")

    # ── 5. BLENDER EXPORT ────────────────────────────────────────────────────

    print("\nTriangulating world points...")
    t0 = perf_counter()
    world_points = image_points.triangulate(volume.camera_array)
    timings["triangulation"] = perf_counter() - t0
    print(f"  {len(world_points.df)} points in {timings['triangulation']:.1f}s")

    # Backgrounds only; calibration runs on the source AVIs above. Each camera
    # skips window_lo - lag leading frames so the emitted frame index equals the
    # aligned sync index, keeping the Blender backgrounds honest with the 3D.
    print("\nTranscoding backgrounds (all-intra H.264, temporally aligned)...")
    t0 = perf_counter()
    background_videos = {}
    for cam_id in CAM_IDS:
        dst = output_dir / "videos" / f"cam_{cam_id}.mp4"
        n_frames = _transcode_all_intra(videos[cam_id], dst, skip=window_lo - lags[cam_id], count=n_aligned)
        background_videos[cam_id] = dst
        print(f"  cam{cam_id}: {n_frames} frames -> {dst.name}")
    timings["transcode"] = perf_counter() - t0
    print(f"  done in {timings['transcode']:.1f}s")

    print("\nExporting Blender scene...")
    t0 = perf_counter()
    volume.save(output_dir)
    scene_script = write_blender_scene(
        volume.camera_array,
        world_points,
        output_dir / "capture_volume_scene.py",
        fps=60,  # OpenCap iPhone footage; must match the transcoded backgrounds
        videos=background_videos,
        wireframe=tracker_registry.wireframe_for(TRACKER_KEY),
    )
    timings["blender"] = perf_counter() - t0
    print(f"  {scene_script.with_suffix('.blend')} in {timings['blender']:.1f}s")

    # ── SUMMARY ──────────────────────────────────────────────────────────────

    timings["total"] = perf_counter() - t_total
    print("\nTiming summary:")
    for stage, seconds in timings.items():
        print(f"  {stage:<15} {seconds:6.1f}s")

    print(f"\nOpen the scene:\n  blender {scene_script.with_suffix('.blend')}")


if __name__ == "__main__":
    main()
