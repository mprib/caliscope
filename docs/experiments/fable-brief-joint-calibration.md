<title>Joint Calibration — Fable Brief</title>

# Joint Calibration: Experiment Results and Production Brief

Prepared: 2026-07-04.
Epic branch: `epic/joint-calibration`.
Prior art survey: `specs/joint-intrinsic-extrinsic-ba-roadmap.md`.

## The Finding

Intrinsic calibration is no longer a required step.
The joint bundle adjustment solver recovers focal length and radial distortion from a blind guess (f = image_width/2, k1=0, k2=0) while simultaneously solving camera poses and 3D point positions.

On real outdoor footage with 1-meter ArUco markers, starting from a 40% wrong focal length guess and zero distortion knowledge, the solver achieved 5.8mm rigidity precision on known marker geometry.
That is 0.58% of the marker size.

The user prints markers, places some on the ground, waves a wand through the volume, and presses calibrate.
No charuco board, no intrinsic calibration video, no careful procedure.

## What Was Proved

Four phases of experiments, each building on the last.

### Phase 1a: Free k2 (synthetic)

Added the second radial distortion coefficient (k2) as a free parameter alongside focal length (f) and primary distortion (k1).
9 parameters per camera: [rvec(3), tvec(3), f, k1, k2].

k2 resolves cleanly.
It does not absorb into f or k1.
k1 and k2 trade along a correlation ridge (individual coefficients shift slightly from truth) but total radial distortion correction is preserved.
Reprojection and pose accuracy are identical to the k2-fixed baseline.

15-point characterization grid (f_scale × distortion perturbation): all points converge to the same floor.
One local minimum at f_scale=0.70 with zero distortion perturbation; any small distortion kick escapes it.

### Phase 1b: Noise and bootstrap fragility (synthetic)

Noise degradation is linear from 0.5px to 3.0px.
No cliff, no phase transition.
At 3px noise (high end for real ArUco detection): 0.22 deg rotation error, 8mm translation error.

Bootstrap (PnP initialization) never fails, even at f=500 (64% below truth).
The practical limit is the solver's own bounds: f_guess must be within ~65-180% of truth for the bounds to include the answer.
f = image_width/2 is a safe universal starting point.

Bounds alone suffice.
Priors and regularization are unnecessary.

### Phase 1c: Multi-marker wand (synthetic)

Product workflow scene: two 30cm ArUco markers on a rigid wand (50cm separation) plus four static floor markers between cameras.

Results at 0.5px noise, blind f guess:

| Scene | f error | rotation error | rigidity RMSE |
|-------|---------|----------------|---------------|
| 70-corner charuco | 0.25% | 0.039 deg | — |
| Wand only (8 corners) | 0.39% | 0.088 deg | 0.39mm |
| Wand + 4 static (24 corners) | 0.11% | 0.016 deg | 0.36mm |

The ArUco wand+static scene outperforms the 70-corner charuco on f recovery and pose accuracy.
Static markers anchor the geometry.
Moving markers provide depth variation that makes f observable.

Key findings during development:
- Marker size matters: 30cm markers at ~1m camera distance are the practical minimum. 10cm is too small for reliable triangulation.
- Static marker placement matters: floor markers between cameras work well. Arbitrary floating positions can produce bad triangulation.
- Rigid body composition must apply the wand offset in local frame, not world frame.

### Phase 1d: Real data (#971 outdoor footage)

3 cameras (3840×2160), 8 ArUco markers (1.0m, DICT_4X4_50), markers 0-3 moving, 4-7 static.
11,864 observations.
Cameras had prior charuco-calibrated intrinsics (f ≈ 3050-3400px).

| Configuration | f source | rigidity RMSE |
|---------------|----------|---------------|
| All markers, correct intrinsics | charuco-calibrated | 4.59mm |
| All markers, blind guess | f=1920, k1=0, k2=0 | 5.78mm |
| Static only, correct intrinsics | charuco-calibrated | 17.19mm |
| Static only, blind guess | f=1920, k1=0, k2=0 | 19.07mm |

The solver converges from a 37-44% wrong blind guess on real data.
Moving markers are essential for f observability — static-only achieves 17-19mm versus 4.6-5.8mm with moving markers included.

## What the User Experience Could Be

### Current workflow (requires intrinsic calibration)

1. Print a large charuco board
2. Record intrinsic calibration video for each camera (wave board, fill the frame)
3. Run intrinsic calibration (per-camera, minutes each)
4. Record extrinsic calibration video (all cameras, wave board through volume)
5. Run extrinsic calibration
6. Record motion capture session
7. Track and triangulate

### Proposed workflow (no intrinsic calibration)

1. Print ArUco markers (letter-paper size, ~30cm)
2. Place some markers on the floor between cameras (static reference)
3. Wave a wand (two markers on a stick) through the capture volume
4. Press calibrate
5. Record motion capture session
6. Track and triangulate

Steps 2-4 replace steps 1-5 of the current workflow.
The charuco board is eliminated.
The separate intrinsic calibration step is eliminated.

### What the user sees after calibration

The calibration produced intrinsics (f, k1, k2) and extrinsics (camera poses) simultaneously.
The user needs to trust the result without understanding the math.
Possible quality indicators:

**Distortion visualization.**
Show the undistorted image next to the raw image, or overlay a distortion grid.
The recovered k1 and k2 define the lens model — the user can see whether the correction looks right.
This replaces the intrinsic calibration report (which showed reprojection error on charuco corners).

**Rigidity report.**
The solver knows the true distances between marker corners (they're the constraints).
After optimization, measure how well the triangulated 3D corners match those known distances.
Show per-marker rigidity RMSE, maybe per-frame.
A small number (sub-1% of marker size) means the calibration is good.

**Recovered intrinsics table.**
Show per-camera focal length and distortion coefficients.
If the user did provide intrinsics (optional), show the delta — how much the solver adjusted them.

**3D scene view.**
The existing storyboard/3D viewer can show camera positions and triangulated marker corners.
The user can visually confirm the cameras are where they expect.

## Architecture Decisions for Fable

These are the questions the spec needs to answer.

### 1. How does `experimental_ba.py` become production code?

The current production `CaptureVolume.optimize()` uses normalized-coordinate residuals with fixed intrinsics (pre-undistort, project with identity K).
The experimental optimizer uses pixel-space residuals with free intrinsics (cv2.projectPoints with trial f/k1/k2).

Options:
- Replace the production residual function entirely (pixel-space everywhere)
- Keep both and dispatch based on a flag
- Refactor to a shared structure with pluggable residual functions

The pixel-space approach is strictly more general.
The normalized approach is a special case (f=1, k=0, pre-undistorted).

### 2. Which intrinsic parameters are free?

Currently proven: f (single focal length, fx=fy), k1, k2.
Not yet tested: separate fx/fy, principal point (cx, cy), tangential distortion (p1, p2), k3.

Recommendation: start with {f, k1, k2} as proven. Consider cx/cy later if users report asymmetric lens issues. Keep p1, p2, k3 fixed — they require dense image coverage to observe, which ArUco markers don't provide.

### 3. Fisheye support

The current solver uses `cv2.projectPoints` (Brown-Conrady polynomial model).
Fisheye lenses need `cv2.fisheye.projectPoints` (equidistant model).
`CameraData.fisheye` boolean already exists.
Per-camera dispatch in the residual function is straightforward.

Recommendation: implement as a follow-on task. The architecture supports it by construction. Backlog task `fisheye-joint-ba` is filed.

### 4. When should intrinsics be freed vs fixed?

If the user provides intrinsics (from a prior calibration or import), should the solver refine them or hold them fixed?

Recommendation: always refine by default. The solver can only improve or match the input. Show the delta so the user knows what changed. Provide a "lock intrinsics" option for users who trust their calibration.

### 5. Backward compatibility

Existing charuco-based projects must continue to work. The joint solver should be the default for new calibrations, with charuco projects using the same solver but with charuco-derived constraints instead of ArUco constraints.

The `ConstraintSet.from_charuco_board()` factory (backlog task `unified-scale-via-constraints`) would make both paths use the same constraint mechanism.

### 6. GUI changes

The intrinsic calibration tab becomes optional. If the user has intrinsic videos, they can calibrate them (for better starting intrinsics). If not, the system uses f = image_width/2.

The extrinsic calibration tab gains:
- Recovered intrinsics display (f, k1, k2 per camera, delta from input if applicable)
- Rigidity report (per-marker RMSE)
- Distortion visualization (optional, could be a separate tab or panel)

The "cameras must have intrinsics" gate on the extrinsic tab relaxes to "cameras must have a resolution" (to compute f = width/2).

## Reference Documents

| Document | Path |
|----------|------|
| Epic roadmap | `specs/epic-joint-calibration.md` |
| Prior art survey | `specs/joint-intrinsic-extrinsic-ba-roadmap.md` |
| Phase 1a results | `docs/experiments/phase-1a-free-k2.md` |
| Phase 1c results | `docs/experiments/phase-1c-multi-marker.md` |
| Phase 1d results | `docs/experiments/phase-1d-real-data.md` |
| Experimental optimizer | `src/caliscope/synthetic/experimental_ba.py` |
| Test suite | `tests/synthetic/test_intrinsic_recovery.py` |
| Wand scene factory | `src/caliscope/synthetic/scene_factories.py` |
| D7 session journal | `journal/2026-07-03.md` (session 9) |
| Session 2 journal | `journal/2026-07-04-s2.md` |
