# Epic: Constellation

**Multi-Camera Calibration Workflow Overhaul**

> *Multiple cameras (like stars) working together to map a 3D space*

---

## Quick Orientation

**What this epic delivers:**
- Tab 1: Multi-Camera Processing (2D landmark extraction with live thumbnails)
- Tab 2: Extrinsic Calibration (bundle adjustment + world setup)
- UI polish: Remove emoji indicators, clean up RMSE display
- Cleanup: Remove broken widgets, update to new report structures

**Current phase:** Phase 4.6 (Tab Integration)

**Key files:**
| Purpose | Location |
|---------|----------|
| Pattern to follow | `src/caliscope/gui/cameras_tab_widget.py` |
| Presenter (Phase 2) | `src/caliscope/gui/presenters/multi_camera_processing_presenter.py` |
| Core function (Phase 1) | `src/caliscope/core/process_synchronized_recording.py` |
| Coverage reports (backend) | `src/caliscope/core/coverage_analysis.py` |

**Documentation convention:**
- Specs and plans live in `specs/` with phase-aligned names
- Format: `phase{N}-{short-description}.md` (e.g., `phase1-process-synchronized-recording.md`)
- Delete spec files when phase is complete and merged to main

**Subphase workflow (for each task):**
1. Implement → type check → test
2. **Before commit:** Spawn architect to review (substance vs ceremony, pattern compliance)
3. If changes needed → fix → re-review
4. If good → architect updates spec (mark complete, update "Next task", add notes)
5. Commit — spec should now be cold-start ready for next session

This ensures quality gates and keeps specs current for session handoffs.

**Branch strategy:**
```
main
  └── epic/constellation (this branch)
        ├── feature/phase-0-quick-wins (merged)
        ├── feature/process-synchronized-recording (merged)
        ├── feature/multi-camera-presenter (merged)
        ├── feature/tab-integration  <- Phase 3
        └── ... (merge each to epic, then epic to main when complete)
```

---

## Progress

### Phase 0: Quick Wins ✓
- [x] **0.1** Replace emoji status indicators with styled text colors
  - Camera list: green/red text via QBrush/QColor (Material palette)
  - RMSE on same line: "Port 0 — 0.32px"
- [x] **0.2** Simplify RMSE display in calibration results
  - Removed edge/corner coverage labels
  - Added dotted underline on row labels for tooltip discoverability
- [x] **0.3** Add comprehensive tooltips with plain-language explanations
  - RMSE thresholds, frame count guidelines, coverage importance
  - Foreshortening explanation for orientation diversity
  - Fixed k3 tooltip (was incorrectly describing fisheye model)
  - Reviewed by CV domain expert for technical accuracy

### Phase 1: Core Processing ✓
- [x] **1.1** Create `compute_sync_indices()` for batch frame synchronization
  - New file: `src/caliscope/recording/frame_sync.py`
  - Equivalence test validates against existing sync_index values
- [x] **1.2** Extract video utilities to `recording/video_utils.py`
- [x] **1.3** Create `process_synchronized_recording()` pure function
  - New file: `src/caliscope/core/process_synchronized_recording.py`
  - Progress callbacks, frame data callbacks, CancellationToken support
- [x] **1.4** Add `get_initial_thumbnails()` utility
- [x] **1.5** Add CancellationToken support
- [x] **1.6** Write comprehensive tests (6 tests)
- [x] **1.7** Rename CSV: `frame_time_history.csv` → `frame_timestamps.csv`

**Commits merged to epic/constellation:**
1. `b524623b feat(recording): add compute_sync_indices for batch frame synchronization`
2. `e4a49131 refactor(recording): extract read_video_properties to video_utils`
3. `5cc54584 feat(core): add process_synchronized_recording with thumbnails and cancellation`
4. `8d5c2320 refactor: rename frame_time_history.csv to frame_timestamps.csv`
5. `4d7fa4eb feat: Phase 1 - batch synchronized recording processing`

### Phase 2: Multi-Camera Presenter ✓
- [x] **2.1** Create presenter skeleton with computed state machine
  - New file: `src/caliscope/gui/presenters/multi_camera_processing_presenter.py`
  - States: UNCONFIGURED → READY → PROCESSING → COMPLETE
- [x] **2.2** Add TaskManager integration
  - `start_processing()`, `cancel_processing()`, `reset()`, `cleanup()`
- [x] **2.3** Add rotation control
  - `set_rotation()` method, `rotation_changed` signal
- [x] **2.4** Add thumbnail preview
  - `_load_initial_thumbnails()`, `_refresh_thumbnail()`, `thumbnail_updated` signal
- [x] **2.5** Add unit tests (12 canary tests)

**Commits merged to epic/constellation:**
1. `b033caae feat(gui): add MultiCameraProcessingPresenter skeleton`
2. `714a4ba7 feat(gui): add TaskManager integration to MultiCameraProcessingPresenter`
3. `4e865cb7 feat(gui): add rotation control to MultiCameraProcessingPresenter`
4. `f01fd406 feat(gui): add thumbnail preview to MultiCameraProcessingPresenter`
5. `5e659760 test: add unit tests for MultiCameraProcessingPresenter`
6. `6cbff8c4 feat: Phase 2 - MultiCameraProcessingPresenter`

### Phase 3: Tab Integration (2-3 hrs)
- [x] **3.1** Create `MultiCameraProcessingWidget` (the View)
  - New file: `src/caliscope/gui/views/multi_camera_processing_widget.py`
  - Camera grid with thumbnails, rotation controls
  - Progress bar, Start/Cancel buttons
  - Coverage summary display
- [x] **3.2** Create `MultiCameraProcessingTab` (the glue layer)
  - New file: `src/caliscope/gui/multi_camera_processing_tab.py`
  - Creates presenter via factory on coordinator
  - Wires signals: Presenter ↔ View, Presenter → Coordinator
  - Handles tab enable/disable based on prerequisites
- [x] **3.3** Add factory method to WorkspaceCoordinator
  - `create_multi_camera_presenter()` method
  - Inject dependencies: task_manager, charuco_tracker
- [x] **3.4** Wire Tab 1 completion to Tab 2 enable
  - On `processing_complete`: persist ImagePoints, enable Tab 2

### Phase 3.5: Interactive UI Refinement ✓
Iterative feedback loop using `scripts/widget_visualization/wv_multi_camera_tab.py`.

**Completed refinements:**
- [x] Larger thumbnails (280px), bigger landmark points (5px)
- [x] Scroll area + dynamic columns for camera grid
- [x] Subsample control spinbox
- [x] Coverage tooltips with dotted underline pattern
- [x] Coverage matrix shows lower triangle only

### Phase 4: Extrinsic Calibration Tab ← CURRENT
- [x] **4.1** Presenter skeleton
- [x] **4.2** Calibration workflow implementation
- [x] **4.3** Transform operations (rotate, filter, align)
- [x] **4.4** Quality panel widget + scale accuracy
- [x] **4.5** View assembly (ExtrinsicCalibrationView with all controls)
- [ ] **4.6** Tab integration (wire tab to use new presenter/view)
- [ ] **4.7** Legacy removal (delete ExtrinsicCalibrationWidget)

**Enhancements beyond original spec:**
- Coverage heatmap in floating dialog (handles any camera count)
- Initial coverage shown before calibration runs
- Coverage updates after filtering operations
- MIN_CELL_SIZE=35px for readability at any camera count

### Phase 5: Cleanup (2-3 hrs)
- [ ] **5.1** Remove broken widgets
  - `SyncedFramesDisplay`, `ExtrinsicPlaybackWidget`, `FrameDictionaryEmitter`
- [ ] **5.2** Update documentation
- [ ] **5.3** Final test suite run

---

## Design Decisions

### Two-Tab Split
- **Tab 1 (Processing):** 2D extraction only → outputs ImagePoints + ExtrinsicCoverageReport
- **Tab 2 (Calibration):** Triangulation + bundle adjustment → outputs calibrated CameraArray
- Rationale: User can assess coverage quality before committing to calibration

### State Machine (Tab 1)
```
UNCONFIGURED → READY → PROCESSING → COMPLETE
     ↑                                  │
     └──────────────────────────────────┘
                  (reset)
```
State is *computed* from internal reality, never stored separately.

### Camera Rotation
Stored on `CameraData.rotation_count` (0, 1, 2, 3 = 90° increments).
Already persists via `CameraArrayRepository`. Presenter modifies, coordinator saves.

### Thumbnail Updates
Display queue pattern from `IntrinsicCalibrationPresenter`. Throttled to ~10 FPS.

### Broken Widgets (Replace, Don't Fix)
- `SyncedFramesDisplay`: Wrong architecture (takes infrastructure, not presenter)
- `ExtrinsicPlaybackWidget`: References non-existent attributes

---

## Session Log

### 2026-01-26: Phase 4.5 Enhancement - Coverage Matrix
- Implemented coverage matrix visualization for ExtrinsicCalibrationView
  - Initial approach: embedded widget beside QualityPanel
  - Final approach: floating dialog via "View Coverage" button (handles 11+ cameras)
  - Coverage shown immediately on load (before calibration)
  - Updates after filtering operations
- Coverage heatmap widget enhanced with MIN_CELL_SIZE=35px for readability
- Scale accuracy signal chain already working from previous session
- Layout: Filter controls as single line with coverage button at end
- **Remaining:** 4.6 (tab integration) and 4.7 (legacy removal) still pending
- Branch: `feature/phase4-extrinsic-calibration-tab`

### 2026-01-25: Phase 3.5 Round 2
- Implemented UI polish: larger thumbnails (280px), bigger points (5px radius)
- Added scroll area + dynamic columns for camera grid responsiveness
- Added subsample control spinbox with proper state management
- Added tooltips for coverage metrics (dotted underline pattern)
- Coverage matrix now shows lower triangle only
- Fixed test script anti-pattern (was overriding button handler instead of configuring spinbox)
- **BLOCKED:** Coverage thresholds miscalibrated - needs CV engineer review
- **BLOCKED:** Landmark points appear in wrong location despite rotation fix
- **BLOCKED:** Missing MESH topology classification logic
- See `specs/phase3-tab-integration.md` for detailed issue descriptions

### 2026-01-25: Phase 3 Complete
- Completed all 4 Phase 3 tasks
- Created `MultiCameraProcessingTab` (glue layer):
  - Creates presenter via coordinator factory
  - Configures with extrinsic_dir and cameras
  - Wires rotation_changed → coordinator.persist_camera_rotation
  - Wires processing_complete → persist ImagePoints → emit signal
  - Handles charuco_changed by recreating presenter with fresh tracker
  - Proper cleanup() method for lifecycle management
- Added `extrinsic_image_points_ready` signal to coordinator
- Wired MainWidget:
  - Multi-Camera tab inserted between Cameras and Capture Volume
  - Tab enabled when: intrinsics calibrated AND extrinsic videos available
  - On processing complete → Capture Volume tab enabled
  - Cleanup chain includes multi_camera_tab
- Tab is named "Multi-Camera" per user preference

### 2026-01-25: Phase 2 Complete
- Completed all 5 Phase 2 tasks
- MultiCameraProcessingPresenter fully implemented with:
  - Computed state machine (UNCONFIGURED/READY/PROCESSING/COMPLETE)
  - TaskManager integration for background processing
  - Rotation control with persistence signaling
  - Thumbnail loading and refresh
  - 12 unit tests covering all behavior
- Merged `feature/multi-camera-presenter` into `epic/constellation`

### 2026-01-25: Phase 1 Complete
- Completed batch sync algorithm with equivalence testing
- Created process_synchronized_recording pure function
- Migrated CSV naming: frame_time_history.csv → frame_timestamps.csv
- Merged `feature/process-synchronized-recording` into `epic/constellation`

### 2026-01-25: Phase 0 Complete
- Replaced emoji indicators with styled text colors (green/red)
- Added comprehensive tooltips with dotted underline discoverability pattern
- Removed edge/corner coverage clutter from intrinsic results display
- Tooltip language reviewed by UX agent (clarity) and CV engineer (accuracy)
- Merged `feature/phase-0-quick-wins` into `epic/constellation`

### 2026-01-24: Planning Session
- Created architecture plan for multi-camera processing presenter
- Audited emoji usage in UI (found in camera_list_widget.py, intrinsic_calibration_widget.py)
- Created this roadmap with 22 subtasks across 6 phases
- Created `epic/constellation` branch

---

## Notes

- Each task is 30-90 minutes. Do one thing, commit, take a breath.
- Open questions (decide when you get there):
  - Tab unlock UX: auto-open Tab 2, or "Continue" button?
  - Coverage display: inline or collapsible section?
