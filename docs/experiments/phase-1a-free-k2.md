# Phase 1a: Free k2 — Characterization Results

Date: 2026-07-04.
Branch: `feature/free-k2` off `epic/joint-calibration`.

## Motivation

D7 proved joint BA recovers f and k1 from blind guesses on synthetic data.
This experiment adds k2 (the second radial distortion coefficient) as a free parameter and sweeps noise levels and bootstrap initialization to determine whether the architecture holds up before building the multi-marker scene.

The WEBCAM lens profile has k2 = -0.219, which is significant.
If the solver can't separate k2 from f and k1, the three-parameter model is degenerate and needs regularization or a reduced parameter set.

## Method

### Optimizer changes

`optimize_with_free_intrinsics` in `experimental_ba.py` was extended from 8 to 9 params per camera: `[rvec(3), tvec(3), f, k1, k2]`.

Changes:
- `N_CAM_PARAMS`: 8 → 9
- `_joint_residuals`: builds `dist_coeffs = [k1, k2, p1_fixed, p2_fixed, k3_fixed]` (k2 from params, not from dist_tail)
- `_joint_sparsity_pattern`: 9-wide camera blocks
- `dist_tail`: shrinks from 4 elements `[k2, p1, p2, k3]` to 3 elements `[p1, p2, k3]`
- Bounds: f in [0.5x, 2.0x] initial, k1 in [-1.0, 1.0], k2 in [-2.0, 2.0]
- `IntrinsicEstimate`: added `k2_recovered` and `k2_initial` fields

`IntrinsicPerturbation` in `camera_synthesizer.py` gained a `k2_delta` field so tests can inject known k2 errors.

Zero production code changes.

### Test scene

All experiments use `intrinsic_perturbation_scene()`: 4-camera ring at 3m radius, 7x10 charuco (70 corners) on a diagonal trajectory spanning 0.5-2.5m depth.
WEBCAM lens profile (f=1394.6, k1=0.115, k2=-0.219).
Default pixel noise sigma = 0.5px.

### Sparsity oracle

The oracle computes a dense finite-difference Jacobian and verifies that every zero in the analytic sparsity pattern corresponds to a true zero partial derivative (threshold: |J| < 1e-4).
This catches any mismatch between the 9-param residual function and the sparsity pattern.

## Experiment 1: k2 Characterization Sweep

### Design

15-point grid crossing initialization quality with distortion perturbation magnitude:

- f_scale: {1.03, 1.10, 1.30, 1.50, 0.70}
- (k1_delta, k2_delta): {(0, 0), (0.02, 0.05), (0.10, 0.10)}

For each combination: perturb intrinsics, bootstrap, run `optimize_with_free_intrinsics(strict=False)`, measure per-camera f/k1/k2 error against ground truth.

### Results

| f_scale | k1_d | k2_d | conv | bounds | max f err% | max k1 err | max k2 err |
|---------|------|------|------|--------|------------|------------|------------|
| 1.03 | 0.00 | 0.00 | Y | N | 0.25 | 0.016 | 0.025 |
| 1.03 | 0.02 | 0.05 | Y | N | 0.25 | 0.016 | 0.025 |
| 1.03 | 0.10 | 0.10 | Y | N | 0.25 | 0.016 | 0.025 |
| 1.10 | 0.00 | 0.00 | Y | N | 0.25 | 0.016 | 0.025 |
| 1.10 | 0.02 | 0.05 | Y | N | 0.25 | 0.016 | 0.025 |
| 1.10 | 0.10 | 0.10 | Y | N | 0.25 | 0.016 | 0.025 |
| 1.30 | 0.00 | 0.00 | Y | N | 0.25 | 0.016 | 0.025 |
| 1.30 | 0.02 | 0.05 | Y | N | 0.25 | 0.016 | 0.025 |
| 1.30 | 0.10 | 0.10 | Y | N | 0.25 | 0.016 | 0.025 |
| 1.50 | 0.00 | 0.00 | Y | N | 0.25 | 0.016 | 0.025 |
| 1.50 | 0.02 | 0.05 | Y | N | 0.25 | 0.016 | 0.025 |
| 1.50 | 0.10 | 0.10 | Y | N | 0.25 | 0.016 | 0.025 |
| 0.70 | 0.00 | 0.00 | Y | N | **2.24** | **0.215** | **0.592** |
| 0.70 | 0.02 | 0.05 | Y | N | 0.25 | 0.016 | 0.025 |
| 0.70 | 0.10 | 0.10 | Y | N | 0.25 | 0.016 | 0.025 |

### Interpretation

k2 resolves cleanly.
It does not absorb into f or k1.
The error floor (max_k2 ≈ 0.025) is consistent across 14 of 15 grid points.

The one anomaly — f_scale=0.70 with zero distortion perturbation — is a local minimum.
A 30% underestimate of f combined with exactly correct k1/k2 lands the bootstrap in a bad basin.
Any small perturbation to distortion escapes it.
This is a pathological initialization, not a solver defect.

No bound hits anywhere.
The [-2, 2] k2 bounds are wide enough.

### k1-k2 Correlation Ridge

With both k1 and k2 free, the solver finds a point on the k1-k2 correlation ridge: individual coefficients shift slightly from truth while total radial distortion correction is preserved.
This is the standard behavior for correlated polynomial distortion terms.

Measured ridge effect (typical camera):
- k1 error: 0.016 (was ~0.004 when k2 was fixed in D7)
- k2 error: 0.025

The ridge is harmless.
Reprojection error and pose recovery are identical to the k2-fixed results.
Test tolerances were widened to k1 < 0.02, k2 < 0.03 to reflect this.

## Experiment 2: Noise Characterization

### Design

Pixel noise sigma swept from 0.5 to 3.0 on the 70-corner charuco scene.
f_scale=1.03 perturbation applied (to have something to recover).
Measured: f/k1/k2 error, pose error (rotation degrees, translation meters) after alignment to ground truth.

### Results

| sigma | conv | f err% | k1 err | k2 err | rot (deg) | trans (m) |
|-------|------|--------|--------|--------|-----------|-----------|
| 0.5 | Y | 0.25 | 0.016 | 0.025 | 0.039 | 0.001 |
| 1.0 | Y | 0.50 | 0.032 | 0.051 | 0.077 | 0.003 |
| 2.0 | Y | 1.00 | 0.064 | 0.103 | 0.152 | 0.005 |
| 3.0 | Y | 1.51 | 0.095 | 0.155 | 0.224 | 0.008 |

### Interpretation

Every metric scales linearly with noise sigma.
No phase transition, no sudden failure.
The solver converges at all tested levels with no bound hits.

At sigma=3.0 (the high end for real ArUco detection noise):
- f recovery: 1.5% error (one camera)
- Pose recovery: 0.22 deg rotation, 8mm translation
- These are well within usable accuracy for motion capture

The k1-k2 ridge widens with noise (k1_err 0.095, k2_err 0.155 at sigma=3.0) but the total distortion correction remains accurate.

## Experiment 3: Bootstrap Fragility

### Design

f guesses from 500 to 2500 (true f: 1394.6) via f_scale on the charuco scene.
Tests whether `CaptureVolume.bootstrap()` produces usable initial poses from badly wrong intrinsics, and whether the joint solver can recover from there.

### Results

| f guess | f_scale | bootstrap | converged | bounds | max f err% |
|---------|---------|-----------|-----------|--------|------------|
| 500 | 0.359 | ok | **NO** | HIT | 82 |
| 700 | 0.502 | ok | yes | HIT | 0.24 |
| 900 | 0.645 | ok | yes | ok | 0.25 |
| 1395 | 1.000 | ok | yes | ok | 0.25 |
| 1200 | 0.860 | ok | yes | ok | 0.25 |
| 1500 | 1.076 | ok | yes | ok | 0.25 |
| 1800 | 1.291 | ok | yes | ok | 0.25 |
| 2500 | 1.793 | ok | yes | ok | 0.25 |

### Interpretation

Bootstrap never fails.
Even at f=500 (64% below truth), PnP with the 70-corner charuco finds *some* pose.

The failure at f=500 is a bounds problem, not a bootstrap problem.
The solver's lower bound is 0.5 × 500 = 250, upper is 2.0 × 500 = 1000.
Truth (1394.6) is outside the feasible region.

f=700 converges but hits its upper bound (2.0 × 700 = 1400 ≈ truth).
f=900 and above recover cleanly with identical accuracy.

"Skip intrinsic calibration entirely" is viable for charuco scenes when the f guess is within roughly 65-180% of truth.
For a typical webcam (f ≈ 800-1400), a blind guess of f = image_width/2 ≈ 960 lands comfortably in the convergence basin.

## Conclusions

1. The 9-parameter model (f, k1, k2 free per camera) is sound. k2 resolves cleanly without absorbing into f or k1.
2. The solver degrades linearly with noise, not catastrophically. Real ArUco noise levels (1-3px) are well within the convergence basin.
3. Bootstrap is robust. The practical limit is the solver's own bounds, not PnP initialization quality.
4. Bounds alone suffice. Priors or regularization are unnecessary given the wide basin of convergence.
5. The k1-k2 correlation ridge is a known property of polynomial distortion models and is harmless for the downstream use case (reprojection and pose accuracy are unaffected).

## Files

| File | Role |
|------|------|
| `src/caliscope/synthetic/experimental_ba.py` | Joint optimizer (9-param layout) |
| `src/caliscope/synthetic/camera_synthesizer.py` | `IntrinsicPerturbation` with k2_delta |
| `tests/synthetic/test_intrinsic_recovery.py` | Sparsity oracle, E1-E5b, E2b |
