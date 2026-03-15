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

| Branch | Task ID(s) | Scope | Risk |
|--------|------------|-------|------|
| `feature/charuco-qt-cleanup` | `charuco-qt-import-debt`, `unify-target-save-patterns` | Remove Qt from `core/charuco.py`, unify board image generation to return ndarrays | Low |
| `feature/import-boundary-cleanup` | (part of `optional-gui-extras`) | Fix `logger.py` QtHandler, verify `task_manager/` isolation, add subprocess import boundary test | Medium |
| `feature/optional-gui-packaging` | (part of `optional-gui-extras`) | Split pyproject.toml into core/`[gui]`/`[tracking]` extras, verify headless install | Medium — integration risk |

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

## Failed Approaches (from design sessions)

- `tracker.track(videos)` on Tracker ABC — circular dependency risk
- Two-layer API hiding ImagePoints — breaks checkpoint/reproducibility
- Union-typed `calibrate_intrinsics(source: str | Path | ImagePoints)` — maintenance hazard
- Naive per-point numpy as numba replacement — 1.8x slower; batched approach fixed it
- `__init__.py` re-exports (Option C) — rejected because `__init__.py` has GUI side effects

## Open Questions

- **TOML format** (`camera-toml-format` task): Rodrigues vs 3x3 rotation, metadata separation. Backlog.
- **`extract_image_points` progress callback**: spec says `logging` + optional callback. Rich integration may change this — decide during Phase 4.
- **Charuco board image generation**: OpenCV or PIL to replace QPixmap? Decide during Phase 3.

## Next Steps

1. Create `epic/scriptable-core` branch from `main`
2. Start Phase 1 branches in parallel: `drop-numba`, `drop-pandera`, `fix-frame-skip-label`
3. `drop-numba` is the highest-value first task — prototype exists at `/tmp/bench_batched_triangulate.py`
