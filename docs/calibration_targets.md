# Calibration Targets

Camera calibration in Caliscope requires a physical target with known geometry. The target serves two purposes: determining the intrinsic properties of each camera (optical center, focal length, lens distortion), and establishing the spatial relationship between cameras (extrinsic calibration).

Caliscope supports three types of calibration targets:

| Target | Intrinsic | Extrinsic | Provides Scale |
|--------|-----------|-----------|----------------|
| ChArUco Board | Yes | Yes | Yes |
| Chessboard | Yes | No | No (intrinsic only) |
| ArUco Markers | No | Yes | Yes |

---

## ChArUco Board

The [ChArUco board](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html) is the default and recommended calibration target. It combines a chessboard pattern with embedded ArUco markers, enabling both intrinsic and extrinsic calibration from the same target.


### Configuration Parameters

**Row and Column Count**: The number of squares in each dimension. **These values are NOT interchangeable.** An 8x5 board is not the same as a 5x8 board. If you purchase a pre-printed board, compare it to the preview in the GUI to verify.

**Board Size**: The target dimensions for the printed image. Controls the PNG output size. Does not affect calibration quality.

**ArUco Scale**: The ratio of ArUco marker size to square size (default 0.75). Not exposed in the GUI but editable in `calibration/targets/intrinsic_charuco.toml` or `extrinsic_charuco.toml`.

**Board Inversion**: Swaps black and white to save ink. Caliscope automatically inverts the camera image during detection when enabled.

**Mirror Board**: For surround camera setups, generates a horizontally flipped PNG for printing on the back of the board. When the primary pattern is not found in a frame, Caliscope flips the image and searches again, mapping detected points back to the standard coordinate frame. This allows cameras viewing opposite sides of the board to calibrate against each other.

### Save Options

- **Save Calibration Board PNG**: Generate the primary board image for printing. This file does not need to be stored in any particular location.
- **Save Mirror Board PNG**: Generate the flipped board image for double-sided printing. Save this separately and print it on the reverse side of the primary board.

### Rigidity Constraints (Automatic)

A ChArUco board is a rigid object with known geometry: every pair of chessboard corners sits a fixed distance apart. Caliscope now feeds that geometry into bundle adjustment as rigidity constraints. The optimizer knows the true distances between corners and pulls the reconstruction toward them, which stabilizes the solution and improves scale accuracy.

This happens automatically. There is nothing to configure and no TOML to edit. If you calibrate with a ChArUco board, the constraints are built from the board you already defined.

---

## Chessboard

A standard checkerboard pattern provides a simpler alternative for intrinsic calibration. Chessboards require no specialized printing.

### Configuration Parameters

**Row and Column Count**: The number of squares in each dimension (the same convention as ChArUco boards). Caliscope derives the internal corner count automatically. As with ChArUco boards, these values are not interchangeable.

### Limitations

Chessboards **cannot be used for extrinsic calibration**. Unlike ChArUco boards, which have uniquely identifiable ArUco markers, a chessboard provides no way to track specific points across multiple frames or establish correspondence between different camera views. Use chessboard targets only when you need to calibrate lens distortion and have a separate method for extrinsic calibration.

---

## ArUco Markers

Caliscope supports extrinsic calibration with one or more ArUco markers. Multiple markers with known distances between them provide rigid body constraints that improve calibration accuracy. Markers can be mobile (carried through the volume) or static (fixed in the scene at surveyed positions).

This approach is particularly useful for large capture volumes where a full board would be too small to detect reliably.

### The `aruco_marker_set.toml` File

The marker set is defined in a TOML file in the workspace targets directory. Here is an example with three markers and two distance links:

```toml
dictionary = 1

[[markers]]
id = 0
size_m = 0.05

[[markers]]
id = 1
size_m = 0.05

[[markers]]
id = 5
size_m = 0.05
static = true

[[markers]]
id = 6
size_m = 0.05
static = true

# Corner link: calipers between one corner on each of two markers on a mobile fixture.
[[links]]
marker_a = 0
corner_a = 1
marker_b = 1
corner_b = 3
distance_m = 0.204

# Center link: tape measure between two static wall markers. No corners named.
[[links]]
marker_a = 5
marker_b = 6
distance_m = 0.500
```

A link may only connect two markers of the same kind — both mobile or both static. The solver skips a mixed static/mobile link, so Caliscope rejects it when the file loads rather than let it silently do nothing.

**`dictionary`** — the OpenCV ArUco dictionary, given as an integer. This is the raw `cv2.aruco` constant, not the string name. See the table below for the common values.

**`[[markers]]`** — each entry defines one marker:

- `id` — the marker index within the dictionary
- `size_m` — physical edge length in meters
- `static` — (optional, default false) set to `true` for markers fixed in the scene. Static markers are triangulated once and held constant during optimization.

**`[[links]]`** — each entry is one measured distance between two markers. See the sections below for the two kinds.

### Dictionary IDs

Set `dictionary` to the integer for the ArUco dictionary you printed from. The common values:

| ID | Dictionary |
|----|------------|
| 0  | DICT_4X4_50 |
| 1  | DICT_4X4_100 |
| 2  | DICT_4X4_250 |
| 3  | DICT_4X4_1000 |
| 4  | DICT_5X5_50 |
| 5  | DICT_5X5_100 |
| 6  | DICT_5X5_250 |
| 7  | DICT_5X5_1000 |
| 8  | DICT_6X6_50 |
| 9  | DICT_6X6_100 |
| 10 | DICT_6X6_250 |
| 11 | DICT_6X6_1000 |
| 12 | DICT_7X7_50 |
| 13 | DICT_7X7_100 |
| 14 | DICT_7X7_250 |
| 15 | DICT_7X7_1000 |
| 16 | DICT_ARUCO_ORIGINAL |

The name encodes the marker grid and the count: `DICT_4X4_100` is a 4×4-bit pattern with 100 distinct markers. A smaller count is more robust to misreads, so pick the smallest dictionary that holds all your marker IDs.

### Corner Numbering

Each marker has four corners, numbered 0 through 3. Hold a printed marker so its pattern is upright — the way it looks in the GUI preview, with the `ID:` label reading left to right along the bottom. Then:

```
  0 ──── 1
  │      │
  │      │
  3 ──── 2
```

- **0** = top-left
- **1** = top-right
- **2** = bottom-right
- **3** = bottom-left

The saved marker PNGs (from "Save All PNGs") print each corner's index next to it, so you can read the numbers straight off the paper. Corner 0 is the top-left corner of the upright marker.

### Distance Links

Each `[[links]]` entry records one distance you measured between two markers. Links are optional — you can define none, one, or many. Bundle adjustment uses whatever rigidity you give it. More links pin the geometry more firmly, but even a single link adds scale information.

There are two kinds.

**Corner links** name a specific corner on each marker:

```toml
[[links]]
marker_a = 0
corner_a = 1
marker_b = 1
corner_b = 3
distance_m = 0.204
```

- `marker_a`, `marker_b` — the two marker IDs
- `corner_a`, `corner_b` — the corner index (0–3) on each marker (see Corner Numbering)
- `distance_m` — the measured distance in meters between those two corners

A corner link constrains exactly the one distance you measured, corner to corner. Use corner links for a rigid fixture you can reach with calipers: two markers mounted on the same board or bracket, close enough that you can touch a caliper jaw to each corner. Add as many corner links as you have measurements. Each is independent — there is no requirement that the markers be the same size or face the same way.

**Center links** omit the corner keys, so the distance is measured center to center:

```toml
[[links]]
marker_a = 5
marker_b = 6
distance_m = 0.500
```

- `marker_a`, `marker_b` — the two marker IDs
- `distance_m` — the measured distance in meters between the two marker centers
- (no `corner_a` / `corner_b`)

A marker's center is the middle of its printed square. Use center links for distances too long for calipers: markers taped to a wall or floor across the room, measured with a tape. You sight or measure center to center because a tape cannot reliably hit a specific corner across that distance.

A center link fires only in frames where all four corners of both markers are triangulated, since the center is computed from the four corners.

### The Scale-Anchor Case

A common setup is markers fixed to a wall or floor, far apart, with one tape-measure run between two of them recorded as a center link. This is worth doing, but understand what it buys you.

The per-marker geometry (each marker's own corner spacing) already carries most of the scale information in the reconstruction. The long tape run does not out-vote it. What the tape run adds is an **independent** scale check across a long baseline. If every marker was printed slightly too large or too small — a systematic error from your printer or PDF scaling — every per-marker distance is biased by the same factor, and nothing in the per-marker geometry can notice, because they all agree with each other. A single tape run measured with a real tape does not share that printing error, so it catches the bias the markers hide.

Think of the center link as a second, independent witness to scale, not as the dominant source of it.

### Measurement Uncertainty: `sigma_m`

Every link carries an assumed uncertainty — how much give-or-take is in your measurement. That uncertainty is `sigma_m`, in meters. It is optional. When you leave it out, Caliscope uses a sensible default for the kind of link:

- **2 mm** (`sigma_m = 0.002`) for corner links — caliper class
- **5 mm** (`sigma_m = 0.005`) for center links — tape-measure class

**Omitting `sigma_m` is the right choice for most users.** The defaults match how well most people measure with the corresponding tool.

Here is what the number does. During bundle adjustment the solver tries to honor every link, but links can disagree with each other and with the camera views. `sigma_m` is how it decides who to trust. A small sigma tells the solver "this distance is nearly exact — bend the reconstruction to match it." A large sigma tells it "this is a rough figure — match it only if it costs nothing." So a link with a small sigma pulls harder on the final geometry than a link with a large one.

Override `sigma_m` when your measurement is meaningfully better or worse than the default tool class:

**A machined or 3D-printed fixture with sub-millimeter certainty.** You know the corner spacing from the CAD model, not from a caliper reading.

```toml
[[links]]
marker_a = 0
corner_a = 0
marker_b = 1
corner_b = 0
distance_m = 0.15000
sigma_m = 0.0005
```

The fabricated distance is trustworthy to half a millimeter, so let it pull hard.

**A long tape run with sag and parallax.** A tape stretched across a room sags in the middle, and sighting the endpoints from an angle adds error.

```toml
[[links]]
marker_a = 2
marker_b = 3
distance_m = 3.450
sigma_m = 0.01
```

You believe this to about a centimeter, no better, so weight it accordingly.

**A quick hand measurement you want counted but not trusted.** A figure you eyeballed with a folding rule and want in the solve as a weak vote.

```toml
[[links]]
marker_a = 0
marker_b = 4
distance_m = 0.80
sigma_m = 0.02
```

Two centimeters of give-or-take keeps this link from overriding better data while still contributing.

**What not to do:** do not set `sigma_m` smaller than you can actually measure. An overconfident link — a hand measurement claiming half-millimeter certainty — forces the solver to honor a number that is really wrong. It cannot bend the distance, so it bends the cameras instead, and your calibration gets worse. Set `sigma_m` to your honest measurement error, not your hopes.

### Workflow

1. Create or edit `aruco_marker_set.toml` in the workspace targets directory
2. Click "Save All PNGs" in the GUI to generate printable markers with corner labels (saved to `marker_images/` subfolder)
3. Print the markers. Mount on rigid backing.
4. Measure distances between markers and record them as `[[links]]` — corner links for caliper-range measurements, center links for tape-measure runs. Set `sigma_m` only where your measurement differs from the default tool class.
5. Place static markers in the scene if using any. Record calibration video with markers visible across cameras.
6. Run extraction, then calibrate. Links enter bundle adjustment automatically as rigidity constraints.

### Scripting API

```python
from caliscope.core.aruco_marker import ArucoMarkerSet
from caliscope.core.constraints import ConstraintSet
from caliscope.core.capture_volume import CaptureVolume

marker_set = ArucoMarkerSet.from_toml(toml_path)
constraints = ConstraintSet.from_marker_set(marker_set)
capture_volume = CaptureVolume(
    camera_array, image_points, world_points, constraints=constraints
)
capture_volume.optimize()
```

### Limitations

ArUco markers **cannot be used for intrinsic calibration**. A single marker does not provide enough geometric constraints to estimate lens distortion parameters. Use a ChArUco board or chessboard for intrinsic calibration before proceeding to ArUco-based extrinsic calibration.

### Future Work

Combining a ChArUco board and loose ArUco markers in a single extrinsic calibration is not yet supported. Today an extrinsic calibration uses either a ChArUco board or an ArUco marker set, not both. Mixed targets are planned.

---

## Physical Size and World Scale

Physical target dimensions affect intrinsic and extrinsic calibration differently.

### Intrinsic Calibration: Size-Independent

**Intrinsic calibration does not use physical size.** The lens distortion model depends only on pixel coordinates and their geometric relationships in the image plane. Whether your calibration target is 10 cm or 1 meter across, the intrinsic parameters (focal length in pixels, distortion coefficients, optical center) remain identical. The algorithm only needs to see the same pattern from multiple viewing angles.

This means you can measure and enter the physical size of your target after intrinsic calibration is complete, or even use different-sized boards for intrinsic and extrinsic stages.

### Extrinsic Calibration: Physical Size Defines World Scale

**Extrinsic calibration requires physical size.** The physical dimension you measure and enter for your calibration target becomes the **scale gauge** for the entire 3D reconstruction. All output coordinates are in meters, scaled relative to this measurement.

**The scale chain**:

1. You measure the target with a ruler or calipers (in centimeters)
2. You enter this measurement in the GUI
3. Caliscope converts to meters internally
4. All 3D output coordinates are in meters at that scale

**Warning**: Measurement error propagates silently through the entire system. If you measure your board squares as 2.5 cm but they are actually 2.4 cm, all 3D coordinates will be approximately 4% too large. There is no validation step to catch this error. The calibration will succeed and produce plausible results at the wrong scale.

**Recommendation**: Use calipers for precise measurement. For ChArUco boards, measure corner to corner across several squares and divide to get a single square's edge length. For ArUco markers, measure one side of the marker corner to corner.

---

## Same-as-Intrinsic Option

When using a ChArUco board for both intrinsic and extrinsic calibration, Caliscope provides a "Same as Intrinsic" checkbox that links the two configurations. Enable this option to avoid entering row count, column count, and square size twice. If you need different boards for the two stages (e.g., a small flat board for intrinsic calibration and a large assembled board for extrinsic calibration), leave this option disabled.

---

## Printing and Preparation

**Flatness is critical for intrinsic calibration.** The algorithm assumes detected points lie on a perfect plane. Use rigid backing. Flatness is less critical for extrinsic calibration, where bundle adjustment can absorb small deviations.

**Measure actual printed dimensions.** Even when printing at exact scale, small variations are common. Measure with a ruler or calipers before entering values in the GUI.

**Physical construction:**

- Mount your printed target to rigid backing (cardboard, foam board, or glass)
- Matte finishes reduce glare that can interfere with corner detection
- For double-sided boards, ensure front and back are precisely aligned

You can use different targets for intrinsic and extrinsic calibration. A small flat board works well for intrinsic; a larger board or ArUco marker may be necessary for extrinsic calibration of a large capture volume. Intrinsic calibration only needs to be performed once per camera.
