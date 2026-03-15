# Epic Plan: scriptable-core

## Overview

Make Caliscope's calibration engine usable as a library without the GUI.
Prompted by Go2Kin forking ~2,500 lines rather than `pip install caliscope`.
GitHub Discussion #963 tracks community feedback.

Epic branch: `epic/scriptable-core` (from `main`)

## Companion Documents

- `specs/scriptable-calibration-api.md` — consensus API spec with all design decisions
- `tasks.json` — task graph with acceptance criteria (filter by `theme: scriptable-core`)

## Phase 1: Dependency Cleanup

Three independent branches, no interdependency.
All merge back to `epic/scriptable-core`.

| Branch | Task ID | Scope | Risk |
|--------|---------|-------|------|
| `feature/drop-numba` | `drop-numba` | Replace `triangulate_sync_index` with batched numpy SVD, convert `NumbaDict` → `dict`, remove numba/llvmlite from pyproject.toml | Low — benchmarked, prototype exists |
| `feature/drop-pandera` | `drop-pandera` | Replace `ImagePointSchema`/`WorldPointSchema` with manual validation in `core/point_data.py`, remove pandera from pyproject.toml | Low — 2 schemas, ~10 fields each |
| `feature/fix-frame-skip-label` | `fix-frame-skip-label` | One-line label change in `cameras_tab_widget.py` | Trivial |

## Phase 2: Serialization Refactor + CaptureVolume Rename

Depends on Phase 1 being merged to epic (clean dependency baseline).

| Branch | Task ID | Scope | Risk |
|--------|---------|-------|------|
| `feature/domain-object-serialization` | `domain-object-serialization` | Add `from_toml()`/`to_toml()` to domain objects, update ~56 call sites (16 source + 40 test), thin out persistence.py | Medium — wide blast radius but mechanical |
| `feature/rename-bundle-to-capture-volume` | `rename-bundle-to-capture-volume` | Rename `PointDataBundle` → `CaptureVolume`, `PointDataBundleRepository` → `CaptureVolumeRepository`, update all imports and references | Low — mechanical rename, no behavior change |

Order: serialization first, then rename.
Doing the rename here means `api.py` exports the real class name, not an alias.

## Phase 3: Import Boundary + Packaging

Depends on Phase 2 (serialization methods must exist before packaging split).

Two branches: first clean up Qt imports so core is headless-safe, then split packaging.

| Branch | Scope | Risk |
|--------|-------|------|
| `feature/import-boundary-cleanup` | Commit 1: Remove dead `board_pixmap()` + Qt imports from `core/charuco.py`. Commit 2: Guard `logger.py` QtHandler with `try/except ImportError`, remove `FramePacketStreamer` from `recording/__init__.py` re-exports, update 6 callers to direct imports, add subprocess import boundary test (28 modules). See `specs/phase3-import-boundary-spec.md`. | Low-Medium |
| `feature/optional-gui-packaging` | Split pyproject.toml: base keeps numpy/opencv/pyav/onnxruntime, `[gui]` extra gets PySide6-essentials/pyvista/pyvistaqt. Remove mediapipe entirely (planned deprecation). Guard `__init__.py` QT_API env var. Verify headless install. | Medium — integration risk |

**Packaging decision**: onnxruntime (47 MB) stays in base — the install friction of a separate `[tracking]` extra outweighs the footprint savings. Only PySide6 + 3D viz move to `[gui]`. The headless API works with primitives (`CameraArray`, `CaptureVolume`, `ImagePoints`) and convenience functions; no coordinator needed.

## Phase 4: API Surface

Depends on Phase 3 (imports must be clean before `api.py` can re-export).

| Branch | Task ID | Scope | Risk |
|--------|---------|-------|------|
| `feature/extract-image-points` | (part of `optional-gui-extras`) | Implement `extract_image_points(videos, tracker)` free function + `calibrate_intrinsics` wrapper | Medium |
| `feature/api-module` | (part of `optional-gui-extras`) | Create `caliscope/api.py` with re-exports, `CalibrationError`, `__all__` | Low |
| `feature/rich-terminal-reporting` | `rich-terminal-reporting` | Add rich dep, create `caliscope/reporting.py`, progress bars + report tables | Low |

## Phase 5: Performance + UX

Independent of API work — can overlap with Phase 4.

| Branch | Task ID | Scope | Risk |
|--------|---------|-------|------|
| `feature/charuco-tracking-speedup` | `charuco-tracking-speedup` | ROI bounding box cache from previous frame + mirror orientation hint | Medium |
| `feature/throttle-gui-display` | `throttle-gui-frame-display` | Configurable display rate for both intrinsic and extrinsic processing views | Medium |
| `feature/import-intrinsics` | `import-intrinsics` | GUI pathway to import pre-calibrated camera_array.toml, skip intrinsic calibration | Medium |

## NOT on This Epic

| Task | Reason |
|------|--------|
| `camera-toml-format` | Design question, backlog — doesn't block API |
| `qt3d-replace-vtk` | Large scope, separate concern |
| `CameraData` → `Camera` rename | Mechanical, separate PR after epic |

## Already Done

| Task | Resolution |
|------|------------|
| `port-to-cam-rename` | PR #944 (commit 71d96e77) — codebase uses `cam_id` everywhere |
| `drop-numba` | Merged to `epic/scriptable-core` (2026-03-14). Replaced numba-JIT with two-tier batched numpy SVD: `triangulate_sync_index` (per-frame) + `triangulate_image_points` (bulk). 15.9ms vs 14ms numba baseline. All tests pass. -181 MB install footprint. |
| `drop-pandera` | Merged to `epic/scriptable-core` (2026-03-14). Replaced `ImagePointSchema`/`WorldPointSchema` with `_validate_dataframe()` + column spec dicts. Removed 3 redundant re-validation calls in `persistence.py`. Uses transient `Int64` for null-safe coercion, downcasts to `int64` for numpy compat. -30 MB install footprint. |
| `fix-frame-skip-label` | Merged to `epic/scriptable-core` (2026-03-14). Changed intrinsic tab label from "Frames to skip:" to "Process every" to match extrinsic tab pattern. |
| `domain-object-serialization` | Merged to `epic/scriptable-core` (2026-03-15). Moved serialization from monolithic `persistence.py` (~700 lines) into domain objects with Path-based methods (`from_toml(path)`, `to_toml(path)`, `to_csv(path)`). persistence.py reduced to ~126 lines of atomic write utilities. 38 files changed, net -186 lines. Bug fix: `rotation.any()` → `rotation is not None`. PairedPoseNetwork legacy `to_dict()`/`from_legacy_dict()` replaced with `to_toml()`/`from_toml()`. |
| `rename-bundle-to-capture-volume` | Merged to `epic/scriptable-core` (2026-03-15). Renamed `PointDataBundle` → `CaptureVolume`, `PointDataBundleRepository` → `CaptureVolumeRepository`, plus all variable/signal/method names (`_bundle` → `_capture_volume`, `bundle_changed` → `capture_volume_changed`, etc.). 3 files renamed, 15 files changed. Only "bundle adjustment" (algorithm name) preserved. |
| `import-boundary-cleanup` | Merged to `epic/scriptable-core` (2026-03-15). Removed dead `board_pixmap()` + Qt imports from `core/charuco.py`. Guarded `logger.py` Qt classes with `try/except ImportError`. Removed `FramePacketStreamer`/`create_streamer` from `recording/__init__.py` re-exports, updated 6 callers to direct imports. Added `test_import_boundary.py` (7 subprocess-based boundary tests). 11 files changed. |

## Failed Approaches (from design sessions)

- `tracker.track(videos)` on Tracker ABC — circular dependency risk
- Two-layer API hiding ImagePoints — breaks checkpoint/reproducibility
- Union-typed `calibrate_intrinsics(source: str | Path | ImagePoints)` — maintenance hazard
- Naive per-point numpy as numba replacement — 1.8x slower; batched approach fixed it
- `__init__.py` re-exports (Option C) — rejected because `__init__.py` has GUI side effects

## Open Questions

- **TOML format** (`camera-toml-format` task): Rodrigues vs 3x3 rotation, metadata separation. Backlog.
- **`extract_image_points` progress callback**: spec says `logging` + optional callback. Rich integration may change this — decide during Phase 4.
- ~~**Charuco board image generation**: OpenCV or PIL to replace QPixmap?~~ Resolved — `board_pixmap()` was dead code, deleted. GUI uses `render_charuco_pixmap()` in `gui/utils/`.

## Next Steps

1. ~~Create `epic/scriptable-core` branch from `main`~~ (done)
2. ~~`drop-numba`~~ (done — merged to epic)
3. ~~`drop-pandera`~~ (done — merged to epic)
4. ~~`fix-frame-skip-label`~~ (done — merged to epic)
5. Phase 1 complete.
6. ~~`domain-object-serialization`~~ (done — merged to epic)
7. ~~`rename-bundle-to-capture-volume`~~ (done — merged to epic)
8. Phase 2 complete.
9. ~~`import-boundary-cleanup`~~ (done — merged to epic)
10. Next: Phase 3 — `optional-gui-packaging` (split pyproject.toml, remove mediapipe).
