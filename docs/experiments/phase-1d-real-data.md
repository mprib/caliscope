# Phase 1d: Real Data Validation (#971 Footage)

Date: 2026-07-04.
Branch: `epic/joint-calibration`.

## Data Source

Issue #971 feature request data. Outdoor track-and-field capture.
3 cameras (cam_ids 1, 2, 3), 3840x2160, focal lengths ~3050-3400px.
8 ArUco markers (DICT_4X4_50, 1.0m each): markers 0-3 moving, markers 4-7 static.
11,864 observations after filtering spurious detections.

Data path: `~/caliscope_data/feature_request/calibration/`

## Intrinsics from prior charuco calibration

| Camera | f (px) | k1 | k2 |
|--------|--------|------|------|
| Cam 1 | 3049.5 | -0.063 | 0.181 |
| Cam 2 | 3401.2 | -0.000 | -0.019 |
| Cam 3 | 3097.5 | -0.009 | -0.028 |

These are the "true" intrinsics for this experiment — recovered from a separate charuco
intrinsic calibration step.

## Experiment 1: All markers, correct intrinsics

Bootstrap with correct intrinsics, joint BA with {f, k1, k2} free.

| Metric | Value |
|--------|-------|
| Converged | Yes |
| Hit bounds | No |
| Bootstrap world points | 1600 |
| Pre-optimize rigidity RMSE | 15.37mm |
| Post-optimize rigidity RMSE | 4.59mm |

f moved 10-40px from the charuco-calibrated values. Distortion coefficients also shifted.
This is expected — the joint solver finds a slightly different (possibly better) intrinsic
solution than the separate charuco calibration.

## Experiment 2: All markers, blind f guess

f = 1920 (image_width/2), k1=0, k2=0 for all cameras. 37-44% wrong starting point.

| Camera | true f | recovered f | f error |
|--------|--------|-------------|---------|
| Cam 1 | 3049.5 | 3089.3 | 1.3% |
| Cam 2 | 3401.2 | 3488.4 | 2.6% |
| Cam 3 | 3097.5 | 3078.8 | 0.6% |

Rigidity RMSE: 5.78mm. Converged, no bound hits.

The solver recovers f from a 37-44% wrong blind guess on real data.
Rigidity is slightly worse than with correct intrinsics (5.8mm vs 4.6mm) but
well within usable accuracy for motion capture.

## Experiment 3: Static markers only, correct intrinsics

6,616 observations from markers 4-7 only. 16 world points (4 corners x 4 markers,
all at STATIC_SYNC_INDEX).

| Camera | true f | recovered f | delta |
|--------|--------|-------------|-------|
| Cam 1 | 3049.5 | 3134.5 | +85 |
| Cam 2 | 3401.2 | 3358.8 | -42 |
| Cam 3 | 3097.5 | 3033.1 | -64 |

Rigidity RMSE: 17.19mm.

f drifts 42-85px even from correct intrinsics. Static markers lack depth variation —
the solver can't cleanly separate f from extrinsics.

## Experiment 4: Static markers only, blind f guess

f = 1920, k1=0, k2=0.

| Camera | true f | recovered f | f error |
|--------|--------|-------------|---------|
| Cam 1 | 3049.5 | 3067.7 | 0.6% |
| Cam 2 | 3401.2 | 3304.0 | 2.9% |
| Cam 3 | 3097.5 | 2976.2 | 3.9% |

Rigidity RMSE: 19.07mm.

Surprisingly similar to the correct-intrinsics case. Static markers have limited f
observability regardless of starting point, so the blind guess doesn't hurt much.

## Summary

| Configuration | f source | rigidity (mm) | f recovery |
|---------------|----------|---------------|------------|
| All markers | correct | 4.59 | within 40px |
| All markers | blind | 5.78 | 0.6-2.6% |
| Static only | correct | 17.19 | drifts 42-85px |
| Static only | blind | 19.07 | 0.6-3.9% |

## Conclusions

1. **Joint BA works on real data.** The solver converges from a blind f guess on real ArUco
   detections with real camera noise.
2. **Moving markers are essential for f observability.** Static markers alone give 17-19mm
   rigidity. Adding moving markers drops it to 4.6-5.8mm.
3. **Static markers still help.** They provide camera connectivity and geometric anchoring
   even without depth variation.
4. **The synthetic predictions hold.** Linear degradation with noise, blind f guess works,
   moving + static is stronger than either alone.
5. **5.8mm rigidity from a blind guess on 1.0m outdoor markers** is the headline number
   for the product workflow.

## Files

Data: `~/caliscope_data/feature_request/calibration/`
