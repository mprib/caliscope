# Phase 1c: Multi-Marker Synthetic Scene — Wand Workflow

Date: 2026-07-04.
Branch: `feature/multi-marker-experiment` off `epic/joint-calibration`.

## Motivation

D7 and Phase 1a proved joint BA recovers {f, k1, k2} from charuco boards (70 corners).
The product workflow uses ArUco markers, not charuco.
This experiment tests whether the same solver works on the actual product configuration: two ArUco markers on a rigid wand.

## Scene Configuration

- 4-camera ring, WEBCAM lens (f=1394.6, k1=0.115, k2=-0.219), radius 1.2m
- Two 30cm ArUco markers on a rigid wand, 50cm separation
- Linked via `MarkerLink` with known corner correspondence
- Diagonal trajectory (0.5-3.5m depth) with tumble, 40 frames
- Constraints: 6 intra-marker distances per marker + 4 cross-marker link distances = 16 total

## Static Marker Issue

The original scene included two static wall markers. These were dropped because static marker triangulation fails on 4-corner ArUcos: when all frames' 2D observations collapse to sync_index=-1, the triangulation is poorly conditioned and one corner can land 2m from its siblings. This poisons the entire BA with 180-million-pixel reprojection errors.

The static marker triangulation bug is separate from the joint BA architecture. Filed as a known issue for investigation.

## Experiment 1: Noise Characterization

Pixel noise sigma 0.5-3.0 on the wand scene, f_scale=1.03 perturbation.

| sigma | conv | bounds | f err% | k1 err | k2 err | rot (deg) | trans (m) | rig RMSE (mm) |
|-------|------|--------|--------|--------|--------|-----------|-----------|---------------|
| 0.5 | Y | ok | 0.39 | 0.007 | 0.014 | 0.088 | 0.002 | 0.39 |
| 1.0 | Y | ok | 0.79 | 0.014 | 0.027 | 0.175 | 0.004 | 0.79 |
| 2.0 | Y | ok | 1.60 | 0.028 | 0.054 | 0.351 | 0.009 | 1.58 |
| 3.0 | Y | ok | 2.41 | 0.042 | 0.081 | 0.526 | 0.013 | 2.36 |

Linear degradation across all metrics.
Rigidity RMSE tracks pixel noise nearly 1:1.
No bound hits, no convergence failures through 3px.

## Experiment 2: Blind f + Zero Distortion

f guess from 500 to 2500 (true: 1394.6), k1=0, k2=0. This simulates "skip intrinsic calibration entirely."

| f guess | f_scale | boot | conv | bounds | f err% | rot (deg) | rig (mm) |
|---------|---------|------|------|--------|--------|-----------|----------|
| 500 | 0.359 | ok | Y | HIT | 82 | 120 | 382 |
| 700 | 0.502 | ok | N | HIT | 51 | — | 48 |
| 960 | 0.688 | ok | Y | ok | 0.39 | 0.088 | 0.39 |
| 1200 | 0.860 | ok | Y | ok | 0.39 | 0.088 | 0.39 |
| 1395 | 1.000 | ok | Y | ok | 0.39 | 0.088 | 0.39 |
| 1800 | 1.291 | ok | Y | ok | 0.40 | 0.088 | 0.39 |
| 2500 | 1.793 | ok | Y | ok | 0.40 | 0.088 | 0.39 |

f=960 (image_width/2, 31% wrong) with zero distortion knowledge recovers perfectly.
f=500 and f=700 fail because the solver's bounds exclude truth.
f=960 through f=2500 all converge to the same floor.

The practical rule: f_guess = image_width / 2 is a safe blind guess for any webcam or machine-vision camera.

## Marker Size Finding

The initial experiment used 10cm markers, which failed.
At 1.2m camera distance, 10cm markers subtend only ~130px (7% of sensor width).
The 4 corners are close together, and triangulation is poorly conditioned.

30cm markers subtend ~350px (18% of width) and work reliably.
A 30cm ArUco is approximately letter-paper size — practical for printing.

## Experiment 3: Full Scene (Wand + 4 Static Floor Markers)

After fixing static marker placement (floor, between cameras at 45/135/225/315 degrees,
facing upward), the full scene works and outperforms both wand-only and the 70-corner charuco.

Blind f guess (f=960, k1=0, k2=0):

| sigma | conv | bounds | f err% | k1 err | k2 err | rot (deg) | trans (m) | rig RMSE (mm) |
|-------|------|--------|--------|--------|--------|-----------|-----------|---------------|
| 0.5 | Y | ok | 0.11 | 0.006 | 0.008 | 0.016 | 0.001 | 0.36 |
| 1.0 | Y | ok | 0.21 | 0.011 | 0.015 | 0.032 | 0.001 | 0.72 |
| 2.0 | Y | ok | 0.41 | 0.023 | 0.031 | 0.063 | 0.003 | 1.44 |
| 3.0 | Y | ok | 0.62 | 0.034 | 0.047 | 0.095 | 0.004 | 2.16 |

Comparison at sigma=0.5:

| Scene | f err% | rot (deg) | rigidity (mm) |
|-------|--------|-----------|---------------|
| Charuco (70 corners) | 0.25 | 0.039 | — |
| Wand only (8 corners) | 0.39 | 0.088 | 0.39 |
| Wand + 4 static (24 corners) | 0.11 | 0.016 | 0.36 |

Static markers anchor the geometry. The full ArUco scene (24 corners total) outperforms the
70-corner charuco on f recovery and pose accuracy.

## Static Marker Placement

Initial experiment used arbitrary positions at odd angles. This caused a single corner to
triangulate 2m from its siblings, poisoning the BA. At zero noise, the solver ignores the
garbage — the problem is noise × bad initialization, not bad initialization alone.

Floor markers between cameras (facing upward) triangulate cleanly. The key is placement
relative to cameras: markers should be visible from multiple cameras with good stereo baseline.

## Rigid Body Composition Bug

The original compose order `p.compose(wand_offset)` applied the offset in world space.
Both markers rotated identically but their displacement was fixed along world X, making
them look like synchronized propellers instead of a rigid stick.

Fix: `wand_offset.compose(p)` applies the offset in the wand's local frame. Verified:
constant 50cm corner-to-corner distance, identical orientation at every frame.

## Conclusions

1. **The full product workflow works.** Wand + static floor markers, blind f guess, sub-0.4mm rigidity at 0.5px noise.
2. **Static markers improve accuracy dramatically.** f recovery 3.5x better, pose error 5x better than wand-only.
3. **The ArUco scene outperforms charuco.** 24 well-placed ArUco corners beat 70 charuco corners.
4. **"Skip intrinsic calibration" is viable.** f = image_width/2, k1=0, k2=0 is a safe starting point.
5. **Marker size matters.** 30cm markers at ~1m distance are the minimum. 10cm is too small.
6. **Static marker placement matters.** Floor, between cameras, facing up. Not floating at arbitrary angles.
7. **Rigidity precision scales linearly with detection noise.** 0.5px → 0.36mm, 3px → 2.16mm.

## Files

| File | Role |
|------|------|
| `src/caliscope/synthetic/scene_factories.py` | `wand_scene`, `wand_scene_with_constraints` factories |
| `src/caliscope/synthetic/explorer/explorer_tab.py` | Explorer preset for wand scene |
