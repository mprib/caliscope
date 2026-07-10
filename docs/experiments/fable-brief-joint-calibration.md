<title>Joint Calibration — Fable Brief</title>

# Joint Calibration: Experiment Results and Production Brief

Prepared: 2026-07-04.
Epic branch: `epic/joint-calibration`.
Prior art survey: `specs/joint-intrinsic-extrinsic-ba-roadmap.md`.

## A Note on Creative Latitude

This brief presents experiment results and poses architecture questions.
It is not a locked-down spec.
Fable has creative latitude on UI design, workflow structure, and quality presentation.
If something in this brief seems wrong or suboptimal, push back — the author trusts Fable's taste and expects Fable to improve on these proposals where it sees opportunity.

## Background: How We Got Here

For #971, we built multi-ArUco calibration with static markers and rigidity constraints.
The constraint system locks world scale through known inter-corner distances.
Static markers anchor the scene geometry.
Moving markers provide coverage across the capture volume.

The rigidity precision on #971 data wasn't where we wanted it.
Suspicion: the intrinsic calibration (separate charuco step) was the weak link.
Imperfect intrinsics get baked into the extrinsic solve with no way to correct them.

So Fable dispatched us on a research mission: can the bundle adjustment recover intrinsics jointly with extrinsics?
If so, we eliminate a source of error and simplify the workflow.

We discovered something stronger than expected.
Not only can the solver recover intrinsics — it doesn't even need a prior intrinsic calibration.
A blind guess (f = image_width/2, k1=0, k2=0) is sufficient.
The solver recovers focal length and distortion from the observation geometry alone.

## The Finding

On the #971 real data, starting from a 40% wrong focal length guess and zero distortion knowledge, the solver achieved 5.8mm rigidity precision on 1-meter markers.
That is 0.58% of the marker size.

With charuco-calibrated intrinsics, the same data achieved 15.4mm rigidity before optimization.
The joint solver improved that to 4.6mm — better than the charuco-calibrated starting point.
The joint solver found a better intrinsic solution than the separate charuco calibration.

### Why this matters: before and after

D7 characterization on synthetic data with a 3% focal length error:

| Metric | Fixed-intrinsics BA | Joint BA |
|--------|-------------------|----------|
| Translation error | 14.2mm | 1.3mm |
| Rotation error | 0.80° | 0.03° |
| f recovery | not attempted | < 0.22% error |

The fixed-intrinsic solver has no way to correct a wrong focal length.
The joint solver absorbs it.

## The Core Insight: Depth Variation Makes f Observable

Focal length and camera-to-scene distance are coupled in the projection equation.
A 10% increase in f with a 10% increase in depth produces nearly identical projections for a distant planar target at fixed depth.
Moving markers at varying depths break this coupling because the f-distance ambiguity produces different predictions for near versus far observations.

This is why moving markers are essential.
Static markers alone gave 17-19mm rigidity on the real data.
Adding moving markers dropped it to 4.6-5.8mm.
Scene geometry — specifically depth variation — is the primary defense for f observability, not solver sophistication or robust loss functions.

A lateral-only trajectory (constant depth) cannot resolve f.
The wand must sweep through depth, not just side to side.
The system should report a depth-ratio metric and warn (softly, no hard gate) when depth variation is insufficient.

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
Moving markers provide the depth variation that makes f observable.

Below ~20 corners, constraints are essential.
A single ArUco (4 corners) without constraints drifts to 2.7mm; with constraints it matches the 70-corner charuco floor at 1.3mm.
Constraints are not optional for the ArUco workflow.

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

The joint solver moved f by 10-40px from the charuco-calibrated values and improved rigidity from 15.4mm to 4.6mm.
This suggests the joint solver found a better intrinsic solution than the separate charuco calibration — adapted to the specific observation geometry.

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

**Traffic-light summary.**
Before any detail, a one-line verdict: "Calibration quality: Good (5.8mm on 1.0m markers, 0.58%)" with green/yellow/red color coding based on rigidity as a percentage of marker size.
The casual user reads this and stops.
The expert clicks through to the details below.

**Distortion visualization.**
Two tiers.
MVP: a synthetic distortion grid rendered purely from the recovered f/k1/k2 coefficients — concentric circles showing barrel/pincushion correction.
No video frames needed, lightweight, can go in the quality panel.
Follow-up: a dialog that grabs a frame from each extrinsic video and shows raw versus undistorted side-by-side.

**Rigidity report.**
The solver knows the true distances between marker corners (they're the constraints).
After optimization, measure how well the triangulated 3D corners match those known distances.
Show per-marker rigidity RMSE.
Note: rigidity validates shape preservation. Absolute scale is validated separately through the known marker dimensions.

One visualization idea: a 3D viewer showing the idealized marker geometry overlaid on the recovered geometry, scrubbable across frames or with all frames superimposed.
This is a nice-to-have, not a requirement — Fable should decide if the complexity is justified.

**Recovered intrinsics table.**
Show per-camera focal length and distortion coefficients.
If the user provided intrinsics (optional), show the delta — how much the solver adjusted them.
A contextual note like "Focal length recovered: 3048px (typical for this sensor)" helps users who skipped intrinsic calibration understand what they got.

**Trajectory quality indicator.**
Report the depth ratio (max depth / min depth) across frames.
Warn softly if depth variation is insufficient for f observability.
No hard gate — just a note in the quality report.

## Architecture Decisions for Fable

These are the questions the spec needs to answer.

### 1. One residual function, not two

The current production `CaptureVolume.optimize()` uses normalized-coordinate residuals with fixed intrinsics (pre-undistort, project with identity K).
The experimental optimizer uses pixel-space residuals with free intrinsics (cv2.projectPoints with trial f/k1/k2).

There is currently zero shared code between the two paths.
The pixel-space approach subsumes the normalized approach (fix f/k1/k2 at their initial values and it's equivalent).

The spec should commit to the pixel-space residual as the single production path.
The normalized-coordinate path becomes dead code.
This is the single most important architectural decision.

### 2. Intrinsic write-back is the central integration point

The experimental optimizer recovers f/k1/k2 but does not write them back to `CameraData.matrix` or `CameraData.distortions`.
Every downstream consumer — undistortion, reprojection reports, reconstruction — reads intrinsics from `CameraData`.
If the joint solver refines f from 1920 to 3050 but `CameraData.matrix[0,0]` still says 1920, everything downstream is wrong.

The spec must define:
- When the write-back happens (after optimization? after filter + re-optimize?)
- Whether `CameraData` grows new fields or overwrites `matrix`/`distortions`
- How cx/cy and p1/p2/k3 (held fixed by the solver) are preserved in the written model
- How `CameraArray.update_extrinsic_params()` (currently hardcoded to 6 params per camera) adapts to 9 params

### 3. Default intrinsics synthesis

`CaptureVolume.bootstrap()` requires intrinsics for PnP pose estimation — it raises `CalibrationError` if `matrix is None`.
For the "no intrinsic calibration" workflow, the system must synthesize a guess camera matrix before bootstrap.

The spec should define where this happens.
Options: a `CameraData.with_default_intrinsics(width, height)` factory, a fallback in `bootstrap()`, or GUI-level synthesis when extrinsic videos are loaded.
Camera resolution comes from the extrinsic video headers (PyAV can read this without decoding).

### 4. Which intrinsic parameters are free?

Currently proven: f (single focal length, fx=fy), k1, k2.
Not yet tested: separate fx/fy, principal point (cx, cy), tangential distortion (p1, p2), k3.

Recommendation: start with {f, k1, k2} as proven.
Consider cx/cy later if users report asymmetric lens issues.
Keep p1, p2, k3 fixed — they require dense image coverage to observe, which ArUco markers don't provide.
Cameras with significant tangential distortion (p1/p2) may see worse results than the synthetic predictions suggest.

### 5. When should intrinsics be freed vs fixed?

If the user provides intrinsics (from a prior calibration or import), should the solver refine them or hold them fixed?

Recommendation: refine by default.
Refinement adapts intrinsics to the specific observation geometry, which usually improves calibration quality, but may diverge from intrinsics optimized over dense charuco coverage.
The Phase 1d real data showed the solver improving on charuco-calibrated intrinsics.
Show the delta so the user knows what changed.
Provide an easy-to-find "lock intrinsics" option for users who trust their calibration.

### 6. Charuco as a first-class target

Charuco boards give better subpixel corner refinement than plain ArUco.
Serious users will prefer them for precision work.
Both targets should feed into the same joint solver.

`ConstraintSet.from_charuco_board()` gives charuco users the same rigidity enforcement and joint intrinsic recovery that ArUco users get.
Same solver, same quality reporting, different target geometry.
The factory is small — charuco geometry is well-defined.

### 7. Filter-then-re-optimize with free intrinsics

The production pipeline does: optimize → filter 2.5% worst → optimize again.
With the joint solver, the second pass should also refine intrinsics.
Should the bounds tighten on the second pass?
The first pass recovers approximate f/k1/k2; tighter bounds around those values would improve stability.

### 8. What happens when bounds are hit?

`JointOptimizationResult.hit_bounds` is tracked but the system doesn't act on it.
Options:
- Warn the user ("focal length recovery hit solver bounds — consider providing intrinsic calibration")
- Auto-widen bounds and re-run
- Fall back to fixed-intrinsic optimization

This is a UX decision with architecture implications.

### 9. Fisheye support

The current solver uses `cv2.projectPoints` (Brown-Conrady polynomial model).
Fisheye lenses (>120° FOV) need `cv2.fisheye.projectPoints` (equidistant model).
`CameraData.fisheye` boolean already exists.

The API dispatch is simple but the models use fundamentally different distortion parameterizations.
The {f, k1, k2} parameter set means different things in each model.
Bounds, initial guesses, and convergence basins are all different.
A mixed rig (some standard, some fisheye) needs per-camera parameter interpretation in bound setup, intrinsic extraction, and UI display.

Recommendation: implement as a follow-on task. Backlog task `fisheye-joint-ba` is filed.

### 10. GUI tab structure

The intrinsic calibration tab becomes optional.
The existing tab-gating logic (`all_intrinsics_calibrated()`) must change.

The recommended approach: keep all tabs, relax the intrinsic gate from "required" to "optional enhancement."
The `StepStatus.AVAILABLE` indicator (blue) already exists and means "available but not required."
The extrinsic tab gate changes from "cameras must have intrinsics" to "cameras must have a resolution" (derived from extrinsic video headers).

### 11. Rigidity-based outlier filtering

A prior finding: the current reprojection-error filter achieves only ~68% precision/recall on 5% contamination because BA absorbs outliers into camera poses.
Constraint violations are pose-independent and see the damage BA hides.
A composite score (reprojection error + constraint violation) may separate outliers more cleanly.
This is an open question worth exploring during implementation — the constraint infrastructure is already in place.

### Assumptions and Risks Not Yet Tested

- **2-camera configurations**: untested. The minimum camera count for bootstrap is 2, but convergence basin and accuracy are uncharacterized below 3 cameras.
- **Rolling shutter**: not modeled. Consumer cameras have ~33ms full-frame readout. At moderate wand speeds this is probably below ArUco detection noise, but it's an unverified assumption.
- **Degenerate camera arrangements**: near-collinear cameras or very narrow baselines relative to scene depth are untested.

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
| CV engineer review | Adversarial review, 2026-07-04 (session notes) |
| UX review | Adversarial review, 2026-07-04 (session notes) |
| Senior dev review | Adversarial review, 2026-07-04 (session notes) |
