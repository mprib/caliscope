# EPIC: Chessboard Intrinsic + ArUco Extrinsic Calibration

## Goal

Replace the Charuco-dependent calibration workflow with simpler, more accessible patterns:

- **Chessboard**: Used for intrinsic calibration only. The Milestone 1 research spike confirmed OpenCV does not disambiguate chessboard rotation, so extrinsic uses ArUco exclusively.
- **ArUco marker**: Single marker moved through scene for extrinsic calibration (configured on Multi-Camera tab). Serves as the reliable extrinsic path regardless of spike outcome.

Users print one sheet (chessboard on one side, ArUco marker on the other) for the entire calibration pipeline.

Charuco deprecated from the GUI (keep code, remove from UI).

### Principle: Separate Recordings for Intrinsic and Extrinsic

Intrinsic and extrinsic calibration have fundamentally different data requirements:

- **Intrinsic**: Board fills the frame of a single camera. Close-up, tilted at varied angles, covering all image regions. Goal: constrain lens distortion model. Multi-camera co-visibility is irrelevant.
- **Extrinsic**: Board visible to multiple cameras simultaneously, moved through shared capture volume. Goal: determine camera-to-camera relative poses.

These are opposing requirements. The app must guide users to perform separate recordings. Never combine them.

> Note: For *development testing*, we can opportunistically use the synchronized intrinsic recording for extrinsic experiments, but this is not a user workflow.

## Test Data

Location: `~/OneDrive/mocap/chessboard_aruco/calibration`
- `intrinsic/`: 4 cameras, **9×6 internal corners** (10×7 grid of squares)
- `extrinsic/`: 4 cameras, ArUco 4×4 dictionary, marker ID 0

---

## Git Strategy

Vertical stacks: get intrinsic fully working first, then extrinsic.

```
main
  └── epic/chessboard-aruco-calibration
        │
        │  Stack 1: Intrinsic Chessboard
        ├── feature/chessboard-foundation    (Milestone 1: domain + tracker + spike)
        ├── feature/intrinsic-calibration-ui (Milestone 2: widget + tab integration)
        │
        │  Stack 2: Extrinsic
        ├── feature/aruco-foundation         (Milestone 3: domain + tracker enhancement)
        ├── feature/extrinsic-calibration-ui (Milestone 4: widget + tab integration)
        │
        │  Validation
        └── (end-to-end testing on epic branch, then merge to main)
```

Each milestone = one feature branch = one conversation (~100k orchestrator tokens).

### Documentation Policy

| Document | Scope | Lifespan | Location |
|----------|-------|----------|----------|
| **EPIC spec** | Roadmap, decisions, milestone checklist | Until EPIC merges to main | `specs/chessboard-aruco-calibration.md` |
| **Milestone spec** | Detailed implementation plan for one branch | Created at milestone start, deleted when branch merges to epic | `specs/milestone-N-<name>.md` |
| **Session plan** | Ephemeral handoff doc, references milestone spec | Overwritten each session | Plan file (via `context-handoff` skill) |

**Rules**:
- `specs/` contains at most one EPIC doc + one active milestone spec at a time
- Milestone specs are deleted after branch merge (they served their purpose)
- Session plans reference milestone specs for grounding — don't duplicate their content
- When starting a new milestone, create its spec as the first step before any implementation

### Delegation Model

Orchestrator (Opus) stays clean — delegates everything, absorbs only summaries.

| Agent | Model | Role |
|-------|-------|------|
| **Explore** | sonnet | Codebase search, pattern gathering, "where is X?" |
| **architect** | inherited (4.6) | Design specs, pattern decisions, trade-off analysis |
| **coder** | sonnet | Implementation from approved spec |
| **senior-dev** | inherited (4.6) | Post-implementation code review |
| **classic-cv-engineer** | inherited (4.6) | Mathematical correctness for calibration algorithms |
| **pyside6-ui-ux** | sonnet | Widget implementation, visual testing |

**Per-milestone workflow:**
1. **Explore** (sonnet) → gather existing patterns, analogs, integration points
2. **architect** (opus) → draft milestone spec with file-level detail
3. User reviews → approve or iterate
4. **coder** (sonnet) → implement from spec (parallel agents for independent files)
5. **senior-dev** (opus) → review implementation
6. Fix issues → commit → merge feature → epic

**Rules:**
- Orchestrator never implements — it synthesizes and delegates
- Milestone spec is the contract between architect and coder
- Coder gets spec + relevant file excerpts, not "go figure it out"
- Failed delegations get fixed, never absorbed into orchestrator

---

## Architectural Decisions

### Pattern ↔ Tracker Separation
- **Patterns** (`Chessboard`, `ArucoMarker`): Immutable frozen dataclasses describing geometry
- **Trackers** (`ChessboardTracker`, `ArucoTracker`): Stateful detection engines that consume patterns
- Composition over inheritance

### "Floating Board" Insight
The bootstrap PnP (`compute_camera_to_object_poses_pnp`) already handles boards that move between frames. It computes camera-to-object pose per sync_index independently. No bundle adjustment changes needed.

### Gauge Setting (Scale + Origin)

Bundle adjustment has 7 degrees of gauge freedom (3 translation, 3 rotation, 1 scale). The physical size of the calibration object fixes the scale gauge.

- **Physical size is irrelevant for intrinsic calibration.** `cv2.calibrateCamera` uses relative corner geometry only. Changing square size has zero effect on focal length, principal point, or distortion. Do NOT show a size field on the intrinsic tab — it confuses users.
- **Physical size is essential for extrinsic calibration.** It sets the world scale. Must be prompted/confirmed on the multi-camera tab.

This means the calibration object config differs by context:
- **Intrinsic tab**: Pattern shape only (rows × columns for chessboard; dictionary + ID for ArUco)
- **Extrinsic tab**: Pattern shape + physical size (square_size_cm or marker_size_cm)

### WorkspaceCoordinator Factory Pattern
- `create_intrinsic_tracker()` → ChessboardTracker or CharucoTracker based on config
- `create_extrinsic_tracker()` → ArucoTracker or CharucoTracker based on config

---

## Milestones

### Milestone 1: Chessboard Foundation
**Branch**: `feature/chessboard-foundation`
**Goal**: Domain layer, working tracker, and disambiguation spike — everything needed before UI work.

**New files**:
- `src/caliscope/core/chessboard.py` — frozen dataclass
- `src/caliscope/trackers/chessboard_tracker.py`
- `src/caliscope/repositories/chessboard_repository.py`

**Modify**:
- `src/caliscope/persistence.py` — add save/load for Chessboard
- `src/caliscope/trackers/tracker_enum.py` — add CHESSBOARD entry

**Deliverables**:

Naming cleanup:
- [x] Rename `frametimes.csv` → `timestamps.csv` convention throughout codebase
- [x] Update all code references (recording, synchronization, workspace guide)

Domain layer:
- [x] `Chessboard(rows, columns, square_size_cm)` frozen dataclass
- [x] `get_object_points()` → (N, 3) array of corner positions in board frame
- [x] TOML persistence round-trip (save/load)
- [x] `ChessboardRepository` following `CharucoRepository` pattern (Repository-SSOT pattern)
- [x] Unit tests for dataclass, object points, and persistence (10 tests passing)

Coordinator integration (Repository-SSOT pattern — new for this EPIC):
- [x] `WorkspaceGuide`: `calibration_dir` and `chessboard_toml` path definitions
- [x] `WorkspaceCoordinator`: `ChessboardRepository`, `chessboard_changed` signal, `update_chessboard()`, `create_intrinsic_tracker()` factory
- [x] Coordinator does NOT cache `self.chessboard` — presenters load from repository

Tracker:
- [x] `ChessboardTracker(chessboard)` implementing `Tracker` ABC
- [x] Uses `cv2.findChessboardCorners()` + `cv2.cornerSubPix()`
- [x] Returns `PointPacket` with `point_id` (0 to N-1 row-major) and `obj_loc` populated
- [x] Implement all required ABC methods (`get_point_name`, `scatter_draw_instructions`)
- [x] Visual verification: `__main__` debug harness in test_chessboard.py overlays detected corners with IDs
- [x] Unit tests (type check clean, all passing)

Test data extraction (from `~/OneDrive/mocap/chessboard_aruco/calibration/`):
- [x] Extract representative frames as PNGs from intrinsic videos
- [x] Include: board-detected frames (various positions/angles), no-board frame, synchronized 4-camera frames
- [x] Store in `tests/sessions/chessboard_intrinsic/` — 8 PNGs (~7MB total)

Research spike — chessboard extrinsic disambiguation:
- [x] **Detection**: All 54 corners found consistently on test frames
- [x] **No-board**: Empty PointPacket returned when board not visible
- [x] **Cross-camera**: 4 cameras at same timestamp → consistent point_id ordering (relies on user-configured rotation_count)
- [x] **Finding**: OpenCV does NOT disambiguate 180° rotation even with asymmetric 9×6 board. Neither `findChessboardCorners` nor `findChessboardCornersSB` re-orders corners.
- [x] **Decision**: Chessboard for intrinsic only. ArUco for extrinsic (inherently unambiguous).

**Status**: **Complete** — all deliverables done, 13 tests passing, 0 type errors

---

### Milestone 2: Intrinsic Calibration UI Integration
**Branch**: `feature/intrinsic-calibration-ui`
**Goal**: User can configure a chessboard, see a visual preview, and run intrinsic calibration end-to-end.

**New files**:
- `src/caliscope/gui/widgets/calibration_object_selector.py` — dropdown + stacked widget (reusable)
- `src/caliscope/gui/widgets/chessboard_config_panel.py` — shape-only config (rows, columns, NO physical size)

**Modify**:
- `src/caliscope/workspace_coordinator.py` (coordinator integration completed in Milestone 1 via Repository-SSOT pattern)
- Cameras tab view
  - Replace CharucoConfigPanel with CalibrationObjectSelector (intrinsic mode)
  - Visual preview of configured chessboard pattern
  - Wire config changes → coordinator → tracker hot-swap
- Intrinsic calibration presenter/processing
  - Frame skipping support (default every Nth frame, matching multi-camera pattern)

**Deliverables**:
- [ ] `CalibrationObjectSelector` widget with dropdown + `QStackedWidget`
  - Intrinsic mode: pattern shape only (no physical size field)
  - Chessboard slot: rows × columns spinboxes
  - (ArUco slot: placeholder or deferred to Milestone 4)
- [ ] Visual preview: rendered chessboard image matching configured dimensions
- [ ] Coordinator wiring: config change → persist → create tracker → emit signal
- [ ] Frame skipping: configurable skip interval for intrinsic processing
- [ ] Manual test: run full intrinsic calibration with chessboard test data
- [ ] Verify camera matrices are reasonable (focal length, distortion coefficients)

**Status**: Not started

---

### Milestone 3: ArUco Foundation
**Branch**: `feature/aruco-foundation`
**Goal**: ArUco domain layer, tracker enhancement with obj_loc, and marker PNG generation.

**New files**:
- `src/caliscope/core/aruco_marker.py` — frozen dataclass
- `src/caliscope/repositories/aruco_marker_repository.py`

**Modify**:
- `src/caliscope/trackers/aruco_tracker.py`
  - Add optional `aruco_marker: ArucoMarker` parameter
  - When provided, populate `obj_loc` with corner positions from marker geometry
  - Verify corner ordering matches OpenCV's `detectMarkers` convention
- `src/caliscope/persistence.py` — add save/load for ArucoMarker

**Deliverables**:

Domain layer:
- [ ] `ArucoMarker(dictionary, marker_id, marker_size_cm)` frozen dataclass
- [ ] `get_corner_positions()` → (4, 3) array relative to marker center
- [ ] `generate_marker_image()` → marker PNG with marker ID printed in small text (non-obscuring)
- [ ] TOML persistence round-trip
- [ ] `ArucoMarkerRepository`
- [ ] Unit tests

Tracker enhancement:
- [ ] `ArucoTracker` accepts optional `ArucoMarker` parameter
- [ ] `obj_loc` populated with known corner geometry when marker provided
- [ ] Corner ordering verified against OpenCV `detectMarkers` convention (TL→TR→BR→BL)
- [ ] Visual verification script: run tracker on extrinsic test video, overlay detected corners with obj_loc
- [ ] Unit tests

**Status**: Not started

---

### Milestone 4: Extrinsic Calibration UI Integration
**Branch**: `feature/extrinsic-calibration-ui`
**Goal**: User can configure extrinsic calibration object (ArUco or chessboard), set physical size (gauge), and run extrinsic calibration end-to-end.

**Modify**:
- `src/caliscope/gui/widgets/calibration_object_selector.py`
  - Extrinsic mode: pattern shape + physical size field (gauge setting)
  - ArUco slot: dictionary dropdown, marker ID, marker size
  - Chessboard slot: not applicable — chessboard is intrinsic-only
- `src/caliscope/gui/widgets/aruco_marker_config_panel.py` (new)
  - Dictionary dropdown, marker ID spinbox, marker size spinbox
  - "Save marker PNG" button
- `src/caliscope/workspace_coordinator.py`
  - Add `aruco_marker` attribute and `ArucoMarkerRepository`
  - Add `update_aruco_marker()` method
  - Add `create_extrinsic_tracker()` factory
  - Emit signal on aruco marker change
- Multi-Camera tab view
  - Add CalibrationObjectSelector (extrinsic mode)
  - Visual preview of configured marker/board
  - Wire config changes → coordinator → tracker

**Deliverables**:
- [ ] `CalibrationObjectSelector` extrinsic mode with physical size (gauge setting)
- [ ] `ArucoMarkerConfigPanel` with dictionary, ID, size, and "Save PNG" button
- [ ] Visual preview of ArUco marker (and chessboard if spike succeeded)
- [ ] Coordinator wiring for extrinsic tracker creation
- [ ] Manual test: run full extrinsic calibration with ArUco test data
- [ ] Verify camera poses are reasonable (relative positions, orientations)

**Status**: Not started

---

### Milestone 5: End-to-End Validation
**Branch**: (work directly on epic branch)
**Goal**: Full pipeline test, fix integration issues, merge to main.

**Deliverables**:
- [ ] Full intrinsic → extrinsic workflow with test data (chessboard → ArUco)
- [ ] If chessboard extrinsic works: test chessboard → chessboard workflow too
- [ ] Verify bundle adjustment converges with ArUco data (4 points per frame)
- [ ] Verify frame skipping produces good calibration with reduced processing time
- [ ] Fix any integration issues discovered
- [ ] Merge epic → main

**Status**: Not started

---

## Resolved Questions

1. **Chessboard dimensions**: 9×6 internal corners (10×7 grid of squares)
2. **ArUco dictionary**: 4×4, marker ID 0
3. **Charuco**: Deprecate from GUI, keep code for now
4. **Pattern selection UX**: Dropdown-driven stacked widget (`CalibrationObjectSelector`), reusable across tabs with mode parameter (intrinsic: shape only; extrinsic: shape + physical size)
5. **Composite point IDs**: Deferred to future EPIC. Approach: separate `object_id` column in ImagePoints + WorldPoints schemas.
6. **Frame skipping**: Intrinsic calibration should support frame skipping (default every Nth frame). Dramatically speeds up processing and stress-tests the automatic frame selector.
7. **Separate recordings**: Intrinsic and extrinsic recordings must always be separate. Different data requirements (fill frame vs. multi-camera co-visibility). App should guide users accordingly.
8. **Gauge setting**: Physical size field only on extrinsic tab. Omit from intrinsic tab to avoid user confusion (size doesn't affect intrinsic math).
9. **Test data strategy**: Extract real frames from test videos as PNGs — no synthetic data generation. Real frames are visually verifiable and can be deliberately broken (e.g., 180° rotation) to test disambiguation. Avoids adding large MP4s to repo.

---

## Future: Multi-Object Tracking (Out of Scope)

Decisions locked in for a future EPIC:

- **Schema change**: Add `object_id: Series[str]` column to both `ImagePointSchema` and `WorldPointSchema`
- **Semantics**: `obj_loc` is always in the **subject's local frame of reference**. Each `object_id` group shares a reference frame. PnP solves camera-to-object pose per object.
- **Example**: Chessboard obj_loc → board frame (origin at corner 0, z=0). ArUco obj_loc → marker frame (origin at center). Body landmarks → no obj_loc (or body root frame).
- **Nothing in this EPIC should prevent** adding `object_id` later — the column is additive.

---

## Reference

Full architectural analysis: `/home/mprib/.claude/plans/ancient-hugging-quilt-agent-abbc879.md`
