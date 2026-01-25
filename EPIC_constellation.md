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

**Current phase:** Phase 1 (Core Processing)

**Key files:**
| Purpose | Location |
|---------|----------|
| Pattern to follow | `src/caliscope/gui/presenters/intrinsic_calibration_presenter.py` |
| Code to extract from | `src/caliscope/managers/synchronized_stream_manager.py` |
| Coverage reports (backend) | `src/caliscope/core/coverage_analysis.py` |

**Documentation convention:**
- Specs and plans live in `specs/` with phase-aligned names
- Format: `phase{N}-{short-description}.md` (e.g., `phase1-process-synchronized-recording.md`)
- Delete spec files when phase is complete and merged to main

**Branch strategy:**
```
main
  └── epic/constellation (this branch)
        ├── feature/phase-0-quick-wins
        ├── feature/process-synchronized-recording
        ├── feature/multi-camera-presenter
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

### Phase 1: Core Processing (3-4 hrs)
- [ ] **1.1** Create `process_synchronized_recording()` skeleton
  - New file: `src/caliscope/core/process_synchronized_recording.py`
  - Simplified batch synchronization (no real-time Synchronizer)
  - Extract logic from `SynchronizedStreamManager.process_streams()`
- [ ] **1.2** Add progress callbacks (`on_progress`, `on_sync_packet`)
  - Integrate with cancellation token
- [ ] **1.3** Test with existing session data
  - Use `tests/sessions/prerecorded_calibration/`

### Phase 2: Multi-Camera Presenter (4-6 hrs)
- [ ] **2.1** Create presenter skeleton with computed state machine
  - New file: `src/caliscope/gui/presenters/multi_camera_processing_presenter.py`
  - States: UNCONFIGURED → READY → PROCESSING → COMPLETE
- [ ] **2.2** Add TaskManager integration
  - `start_processing()`, `cancel_processing()` methods
- [ ] **2.3** Add rotation control and thumbnail preview
  - Use `CameraData.rotation_count` pattern

### Phase 3: Tab Integration (2-3 hrs)
- [ ] **3.1** Create basic widget layout
  - New file: `src/caliscope/gui/views/multi_camera_processing_widget.py`
  - Camera grid, progress bar, Start/Cancel buttons
- [ ] **3.2** Add `CameraThumbnailWidget` with rotation controls
- [ ] **3.3** Create tab and wire to coordinator
  - New file: `src/caliscope/gui/multi_camera_processing_tab.py`

### Phase 4: Extrinsic Calibration Tab (3-4 hrs)
- [ ] **4.1** Review existing `ExtrinsicCalibrationTab`
- [ ] **4.2** Add coverage quality display (heatmap, summary, guidance)
- [ ] **4.3** Add calibration controls (button, progress, RMSE)
- [ ] **4.4** Wire Tab 1 completion to Tab 2 enable

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
