# Spec: Scriptable Calibration API

## Status: Consensus Draft (post-review, with owner annotations)

Four specialist reviews (architect, UX, senior dev, data pipeline) produced convergence on all major decisions.
This spec reflects the final consensus plus owner decisions made during the 2026-03-14 session.

## Problem Statement

External consumers (e.g., Go2Kin/Pose2Sim pipeline) want to use Caliscope's calibration engine without the GUI.
The current codebase makes this impractical:

1. **No video-to-points convenience function** — Caliscope assumes ImagePoints already exist as CSV. Go2Kin wrote ~200 lines to bridge this gap.
2. **Heavy transitive dependencies** — importing any Caliscope module risks pulling in PySide6, PyVista, or mediapipe.
3. **No top-level orchestration** — the extrinsic pipeline requires knowing internal wiring (pose network → triangulate → bundle → optimize).

Go2Kin's response was to **fork ~2,500 lines** of calibration algorithms rather than `pip install caliscope`.
The algorithms were copied nearly verbatim — the math wasn't the problem, the packaging was.

### Evidence

- Go2Kin `code/calibration/` — 14 files, each with `# Adapted from caliscope/...`
- @f_fraysse feedback: video display during calibration wastes resources, heavy deps are friction, no way to save/load intrinsics independently from extrinsics

## Primitives

The scripting API is built on typed primitives.
Each has clear identity and construction semantics.

### `Charuco`

Calibration target definition.
Configuration object — feeds into tracker construction.
The current 10-parameter constructor is GUI-oriented; a simplified classmethod serves scripting users:

```python
charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=3.0)

# Generate printable board images (must work headless — no Qt dependency)
charuco.save_image("charuco_front.png")
charuco.save_mirror_image("charuco_back.png")
```

Board image generation currently uses `QPixmap`/`QImage`.
This must be moved to OpenCV or PIL as part of the import boundary cleanup (Phase 1) so it works without PySide6.

### `Tracker` (ABC)

Detects 2D landmarks (and optionally their known 3D positions) in video frames.
Frame-level contract: `get_points(frame) → PointPacket`.
The Tracker does NOT know about videos — video I/O is handled by `extract_image_points()`.

```python
tracker = CharucoTracker(charuco)
```

Future: ONNX trackers with configured 3D landmark mappings could serve as extrinsic calibration targets without Charuco/ArUco boards.

### `Camera` (currently `CameraData`)

Single camera.
At minimum: `cam_id` and `image_size`.
After intrinsic calibration: `matrix` and `distortions`.
After extrinsic calibration: `rotation` and `translation`.

Properties for state inspection:
- `camera.has_intrinsics` → bool
- `camera.has_extrinsics` → bool

Construction:

```python
# From scratch
cam = Camera(cam_id=0, image_size=(1920, 1080))
cam = Camera(cam_id=0, image_size=(1920, 1080), fisheye=True)

# Individual cameras load from CameraArray, not their own files
cameras = CameraArray.from_toml("camera_array.toml")
cam = cameras[0]
```

No `Camera.from_toml()` — our config format stores arrays of cameras, not individual cameras.
Single-camera access is via `CameraArray[cam_id]`.

### `CameraArray`

Collection of cameras.
Dict-like access via `cameras[cam_id]`.

```python
from pathlib import Path

project_dir = Path("/path/to/my/project")

# From uncalibrated cameras (reads resolution from video metadata via PyAV)
intrinsic_videos = {
    0: project_dir / "calibration" / "intrinsic" / "cam_0.mp4",
    1: project_dir / "calibration" / "intrinsic" / "cam_1.mp4",
    2: project_dir / "calibration" / "intrinsic" / "cam_2.mp4",
}
cameras = CameraArray.from_videos(intrinsic_videos)

# From known image sizes (no video I/O needed)
cameras = CameraArray.from_image_sizes({0: (1920, 1080), 1: (1920, 1080)})

# From saved calibration (intrinsics already done)
cameras = CameraArray.from_toml(project_dir / "camera_array.toml")

# From dict of Camera objects
cameras = CameraArray({0: cam_0, 1: cam_1, 2: cam_2})
```

### `ImagePoints`

2D observations indexed by (sync_index, point_id, cam_id).
Validated DataFrame.
Output of tracking, input to both intrinsic and extrinsic calibration.

```python
# From tracking
points = extract_image_points(videos, tracker)

# From prior extraction
points = ImagePoints.from_csv("image_points.csv")

# Save checkpoint
points.to_csv("image_points.csv")
```

`WorldPoints` is intentionally absent from the public API.
It is an internal intermediate — created inside `CaptureVolume.bootstrap()` via triangulation.
Power users can access it via `capture_volume.world_points` but never need to construct it directly.

### `CaptureVolume` (currently `PointDataBundle`)

The calibrated capture system: CameraArray + ImagePoints + WorldPoints.
Immutable (frozen dataclass).
All refinement methods return new instances.

```python
capture_volume.optimize()                        → CaptureVolume
capture_volume.filter_by_percentile_error(2.5)   → CaptureVolume
capture_volume.filter_by_absolute_error(5.0)     → CaptureVolume
capture_volume.align_to_object(sync_index)       → CaptureVolume
capture_volume.rotate("x", 90)                   → CaptureVolume
```

Save/load writes 3 files to a directory (camera_array.toml + image_points.csv + world_points.csv):

```python
capture_volume.save(project_dir / "capture_volume")
capture_volume = CaptureVolume.load(project_dir / "capture_volume")
```

## Functions

### `extract_image_points(videos, tracker)`

Free function in `caliscope/api.py`.
The #1 missing piece.
Always takes `dict[int, Path | str]` as the first argument — camera ID is always explicit, no filename convention assumptions.
Tracker is the second argument.

Internally: opens video(s) with PyAV (never cv2.VideoCapture), runs tracker frame-by-frame, collects PointPackets, assembles ImagePoints DataFrame.

```python
# Single video (intrinsic use case) — still a dict, cam_id explicit
points = extract_image_points({0: "intrinsic/cam_0.mp4"}, tracker)

# Multiple videos (extrinsic use case)
points = extract_image_points({
    0: "extrinsic/cam_0.mp4",
    1: "extrinsic/cam_1.mp4",
    2: "extrinsic/cam_2.mp4",
}, tracker)
```

NOT a method on Tracker.
The Tracker ABC stays frame-level (`get_points`).
This preserves the dependency boundary: `caliscope/trackers/` does not import PyAV.

### `calibrate_intrinsics(image_points, camera)`

Wraps `run_intrinsic_calibration()`.
Pure function, Qt-free.
Frame selection + `cv2.calibrateCamera`.
Returns `IntrinsicCalibrationOutput` containing both the calibrated camera and a quality report (RMSE, coverage metrics, selected frames).

```python
output = calibrate_intrinsics(points, cameras[cam_id])
cameras[cam_id] = output.camera
print(f"Camera {cam_id}: RMSE={output.report.rmse:.3f}px")
```

Returning the full output (not just Camera) forces awareness of calibration quality.
A 15px RMSE is useless — the user should see it.

### `CaptureVolume.bootstrap(image_points, camera_array)`

Class method orchestrating the full extrinsic pipeline:

1. **Deepcopy** input CameraArray (pose bootstrap mutates in place)
2. `build_paired_pose_network(image_points, camera_array)`
3. `pose_network.apply_to(camera_array)`
4. `image_points.triangulate(camera_array)` → WorldPoints
5. Construct `CaptureVolume(camera_array, image_points, world_points)`

Does NOT auto-optimize.
The user calls `.optimize()` explicitly.

### `CalibrationError`

Single exception class with actionable messages:

```
CalibrationError: Cannot run extrinsic calibration — cameras [2, 4] have
no intrinsic calibration.

Run calibrate_intrinsics() for each camera first:
    output = calibrate_intrinsics(points, cameras[2])
    cameras[2] = output.camera
```

## Pipeline

The scripting API is a series of typed transformations between primitives.
Each function has a clear signature: primitives in, primitive out.

```
Charuco ──→ CharucoTracker (is-a Tracker)

dict[cam_id, Path] + Tracker ──→ ImagePoints
ImagePoints + Camera ──→ IntrinsicCalibrationOutput (.camera, .report)

dict[cam_id, Path] + Tracker ──→ ImagePoints (multi-cam)
ImagePoints + CameraArray ──→ CaptureVolume (via .bootstrap())

CaptureVolume ──→ .optimize() / .filter() / .align_to_object() / .rotate() ──→ CaptureVolume
```

## Complete Workflow Example

```python
from pathlib import Path
from caliscope.api import (
    Charuco,
    CharucoTracker,
    CameraArray,
    CaptureVolume,
    ImagePoints,
    extract_image_points,
    calibrate_intrinsics,
)

project_dir = Path("/path/to/my/project")

# --- Define calibration target and tracker ---
charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=3.0)
tracker = CharucoTracker(charuco)

# --- Initialize cameras from video metadata ---
intrinsic_videos = {
    0: project_dir / "calibration" / "intrinsic" / "cam_0.mp4",
    1: project_dir / "calibration" / "intrinsic" / "cam_1.mp4",
    2: project_dir / "calibration" / "intrinsic" / "cam_2.mp4",
}

cameras = CameraArray.from_videos(intrinsic_videos)

# --- Intrinsic calibration (per camera) ---
for cam_id, video in intrinsic_videos.items():
    points = extract_image_points({cam_id: video}, tracker)
    # CHECKPOINT: points.to_csv(f"checkpoints/intrinsic_cam_{cam_id}.csv")
    output = calibrate_intrinsics(points, cameras[cam_id])
    cameras[cam_id] = output.camera
    print(f"Camera {cam_id}: RMSE={output.report.rmse:.3f}px")

# CHECKPOINT: cameras.to_toml(project_dir / "camera_array.toml")
# RESUME:    cameras = CameraArray.from_toml(project_dir / "camera_array.toml")

# --- Extrinsic calibration ---
extrinsic_videos = {
    0: project_dir / "calibration" / "extrinsic" / "cam_0.mp4",
    1: project_dir / "calibration" / "extrinsic" / "cam_1.mp4",
    2: project_dir / "calibration" / "extrinsic" / "cam_2.mp4",
}

points = extract_image_points(extrinsic_videos, tracker)
# CHECKPOINT: points.to_csv(project_dir / "checkpoints" / "extrinsic_points.csv")

capture_volume = CaptureVolume.bootstrap(points, cameras)

# --- Refine ---
capture_volume = capture_volume.optimize()
capture_volume = capture_volume.filter_by_percentile_error(2.5)
capture_volume = capture_volume.optimize()

# --- Save (writes camera_array.toml + image_points.csv + world_points.csv) ---
capture_volume.save(project_dir / "capture_volume")
# RESUME: capture_volume = CaptureVolume.load(project_dir / "capture_volume")
```

## Workflow — Pre-Calibrated Cameras

```python
# Skip intrinsics — cameras already calibrated last month
cameras = CameraArray.from_toml(project_dir / "camera_array.toml")

points = extract_image_points(extrinsic_videos, tracker)
capture_volume = CaptureVolume.bootstrap(points, cameras)
capture_volume = capture_volume.optimize().filter_by_percentile_error(2.5).optimize()
capture_volume.save(project_dir / "capture_volume")
```

## Public API Surface (`caliscope/api.py`)

| Symbol | Kind | Notes |
|--------|------|-------|
| `Charuco` | class | Add `from_squares()` classmethod |
| `CharucoTracker` | class | Existing |
| `Camera` | alias | For `CameraData` (rename deferred) |
| `CameraArray` | class | Add `__getitem__`/`__setitem__`, `from_videos()`, `from_image_sizes()`, `from_toml()`/`to_toml()` |
| `ImagePoints` | class | Add `to_csv()` |
| `CaptureVolume` | class | Renamed from `PointDataBundle` in Phase 2 |
| `extract_image_points` | function | New — the #1 missing piece |
| `calibrate_intrinsics` | function | Wrapper returning `IntrinsicCalibrationOutput` |
| `CalibrationError` | exception | New — actionable messages |

**Not in public surface**: `WorldPoints` (internal intermediate, accessible via `capture_volume.world_points`), `Tracker` ABC (import from `caliscope.trackers` for custom implementations), `PointPacket` (frame-level internal type).

## Serialization

Methods on domain objects, delegating to persistence helpers internally.
Clean cut — no standalone `persistence.load_*`/`save_*` functions in public API.
See task `domain-object-serialization`.

| Type | Save | Load |
|------|------|------|
| `ImagePoints` | `points.to_csv(path)` | `ImagePoints.from_csv(path)` |
| `CameraArray` | `cameras.to_toml(path)` | `CameraArray.from_toml(path)` |
| `CaptureVolume` | `capture_volume.save(dir)` | `CaptureVolume.load(dir)` |

Optional provenance sidecar: `capture_volume.save(dir, provenance={...})` writes a `provenance.toml` alongside the bundle files.
CaptureVolume stays a clean frozen dataclass.

### TOML Format — NEEDS CLARIFICATION

**Current caliscope format issues:**
- Rotation stored as 3x3 matrix — should be 3x1 Rodrigues vector (more compact, matches what cv2.solvePnP returns, matches aniposelib convention)
- Extra GUI metadata fields (`exposure`, `grid_count`, `rotation_count`, `verified_resolutions`, `ignore`) mixed in with calibration math
- Aniposelib export exists as a separate function (`save_camera_array_aniposelib`) with a different structure (top-level `[cam_N]` sections, Rodrigues rotation, no GUI metadata)

**Questions to resolve (separate task):**
1. Should we switch caliscope's native format to Rodrigues rotation?
2. Should GUI metadata live in a separate section or file?
3. Can we converge with aniposelib format, or keep two formats with explicit export?
4. Is there a community-standard camera calibration TOML/JSON format emerging that we should align with?

This is scoped as a separate task — the scripting API works with whatever format exists, as long as `from_toml()`/`to_toml()` are symmetric.

## Design Decisions (Settled)

1. **`extract_image_points` is a free function** — not on Tracker ABC. Tracker stays frame-level, no circular dependency risk.

2. **`extract_image_points` takes `dict[int, Path]` first, tracker second** — videos are the data, tracker is the method. cam_id is always explicit. Single-camera case: `{0: "cam_0.mp4"}`.

3. **`calibrate_intrinsics` returns `IntrinsicCalibrationOutput`** — camera + report. Forces awareness of calibration quality.

4. **`CaptureVolume.bootstrap()` deepcopies input CameraArray** — pose bootstrap mutates in place, callers must not be surprised.

5. **`.bootstrap()` does NOT auto-optimize** — user calls `.optimize()` explicitly.

6. **`optimize()` defaults to `verbose=0`** — library, not application. Opt-in for scipy output.

7. **Save/load on domain objects** — clean cut from persistence.py standalone functions. Repositories stay (manage workspace paths, signals, coordination) but call domain methods instead of persistence functions.

8. **Drop pandera** — replace with hand-rolled validation. Spend the dependency budget on rich instead for terminal reporting.

9. **`PointDataBundle` → `CaptureVolume` rename on epic** — done in Phase 2 alongside serialization refactor so `api.py` exports the real class, not an alias. `CameraData` → `Camera` rename still deferred (separate PR).

10. **Drop numba entirely** — benchmarked on real calibration data (2,175 observations, 660 points). Batched numpy SVD (grouping points by view count, 3 SVD calls instead of 660) matches numba performance (~15ms vs ~14ms) with identical numerical results (max diff 7.8e-16). Removes 181 MB of dependencies (numba + llvmlite).

11. **PyAV for all video I/O** — never cv2.VideoCapture. Non-negotiable per project standards.

12. **Rich for terminal reporting** — progress bars, calibration report tables, colored output. Replaces pandera in the dependency budget.

13. **onnxruntime as `[tracking]` extra** — only used by ONNX tracker, not needed for calibration. 47 MB saved for core install.

14. **No `Camera.from_toml()`** — our config format stores camera arrays, not individual cameras. Single camera access via `CameraArray[cam_id]`.

## Implementation Phases

### Phase 0: Prerequisite Refactors

- **`domain-object-serialization`** — Move persistence.py functions to domain object methods. Repositories stay, call domain methods.
- **`drop-numba`** — Replace numba-JIT triangulation with batched numpy SVD.
- **`drop-pandera`** — Replace with hand-rolled validation.

### Phase 1: Import Boundary Cleanup

Fix module-level Qt imports that leak outside `gui/`:

| File | Problem | Fix |
|------|---------|-----|
| `core/charuco.py` | `QPixmap`, `QImage` at module level | Move board rendering to OpenCV/PIL so `save_image()` works headless |
| `logger.py` | `QtHandler()` instantiated at module level | Lazy-init when GUI is running |
| `task_manager/` | `QObject`, `QThread` | Verify no transitive pull from scripting path |

### Phase 2: Packaging

Split `pyproject.toml` into core vs `caliscope[gui]` extras.
Verify `pip install caliscope` works without PySide6/PyVista.

Core (~288 MB): numpy, scipy, opencv, pandas, PyAV, rich, rtoml
GUI extra: PySide6, pyvistaqt, pyvista, PyOpenGL, pyqtgraph
Tracking extra: onnxruntime

### Phase 3: Core API Functions

- `extract_image_points(videos, tracker)` — the #1 missing piece
- `CaptureVolume.bootstrap(points, cameras)` — with deepcopy guard
- `calibrate_intrinsics` wrapper in `caliscope/api.py`
- `Charuco.from_squares()` classmethod
- `CameraArray.__getitem__`/`__setitem__`
- `CalibrationError` with actionable messages

### Phase 4: Public Module

- Create `caliscope/api.py` with re-exports and aliases
- `Camera = CameraData`, `CaptureVolume = PointDataBundle`

### Phase 5: Remaining Renames (separate PR)

- `CameraData` → `Camera` (mechanical, codebase-wide — not on this epic)
- ~~`PointDataBundle` → `CaptureVolume`~~ — moved to Phase 2, done alongside serialization refactor
