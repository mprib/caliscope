# The ArUco Calibration Set

Extrinsic calibration with ArUco markers is configured through a **marker set**: which markers exist, how large they are, which are fixed in the scene, and what distances you measured between them.
The set is defined in `aruco_marker_set.toml` in the workspace targets directory (`calibration/targets/`).
This page covers the concepts behind a good marker set, the TOML format, and the GUI panel.

If you are deciding *whether* to use ArUco markers rather than a ChArUco board, start with [Calibration Targets](calibration_targets.md).
If you are calibrating without prior intrinsic calibration, the marker set carries extra weight; see [skipping intrinsic calibration](extrinsic_calibration.md#skipping-intrinsic-calibration).

## Rigidity Constraints

A ChArUco board hands the solver about 70 uniquely identified corners per frame, and that density alone stabilizes the solution.
A plain ArUco scene gives four corners per marker.
With so few points, rigidity constraints are what hold the solve together: in synthetic testing, a single marker solved with no constraints at all drifted to 2.7 mm of error, while the same marker with its constraints matched a 70-corner ChArUco board at 1.3 mm.

**The baseline constraints are automatic.** Each marker is a square of known size, so Caliscope builds the full corner-to-corner distance truss for every marker from its `size_m`.
You do not configure this; it is derived from the marker definition.
It also means `size_m` does double duty — it is both the scale gauge for the reconstruction and the source of each marker's rigidity — so measure the printed marker with calipers rather than trusting the PDF.

Everything else on this page adds constraints beyond that baseline:

- **Static markers** anchor the scene geometry.
  In the same synthetic testing, a two-marker wand plus four static floor markers cut focal-length recovery error roughly 3x versus the wand alone, and outperformed the 70-corner ChArUco board.
- **Distance links** tie separate markers together and add an independent check on world scale (see [The Scale-Anchor Case](#the-scale-anchor-case)).
- **Mirror pairs** let cameras on opposite sides of a double-sided board calibrate against each other.

None of these additions is mandatory.
But a minimal set — one small mobile marker and nothing else — sits at the bottom of what the solver can hold.
The strongest practical layout pairs static anchors with a mobile marker swept through the volume.

## Static and Mobile Markers

Every marker in the set is either **mobile** (the default) or **static** (`static = true`).
The distinction runs through everything else on this page.

**Static markers** are fixed in the scene: taped to a wall, bolted to a floor plate.
Caliscope treats a static marker's corners as one set of 3D points shared across every frame, so every observation from every camera and every moment pins the same four points.
This anchors the reconstruction.

**Mobile markers** are carried through the volume.
They do more than fill space: a mobile marker swept *toward and away from* the cameras supplies the depth variation that makes focal length observable.
If you skip intrinsic calibration or let the solver refine intrinsics, this depth sweep is what the solver feeds on.
See [the depth-ratio gate](extrinsic_calibration.md#the-depth-ratio-gate) for why a flat, constant-depth sweep is not enough.

Two rules follow from the distinction:

1. **Links and mirror pairs must join same-class markers** (static–static or mobile–mobile).
   A distance between a fixed wall marker and a marker in your hand changes every frame, so the solver cannot use it as a rigid constraint.
   Caliscope rejects a mixed link with an error when the TOML loads.
2. **A static marker that moves gets dropped.** Before optimization, Caliscope checks each static marker's rigidity against the bootstrap reconstruction.
   A marker whose apparent corner geometry is badly inconsistent across frames (it was bumped, or it was never actually fixed) is excluded from the solve, with a warning in the log, and calibration continues without it.

## Editing the Set

The TOML file is the editing interface.
The GUI shows a read-only summary.

In the GUI, select **ArUco** as the extrinsic target type on the Project tab.
The panel displays the loaded markers (count, IDs, sizes) and three buttons:

- **Edit TOML**: opens `aruco_marker_set.toml` in your text editor
- **Reload**: re-reads the file after you edit it
- **Save All PNGs**: writes one printable PNG per marker to a `marker_images/` subfolder, with corner indices labeled

Edit the file, save, and click Reload.
If the file has an error (a mixed static/mobile link, a duplicate marker ID, a marker ID that exceeds the dictionary), the reload fails and the log states why.

## The `aruco_marker_set.toml` File

Here is an example with three mobile markers, two static markers, and two distance links:

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

**`dictionary`**: the OpenCV ArUco dictionary, given as an integer.
This is the raw `cv2.aruco` constant, not the string name.
See the table below for the common values.

**`[[markers]]`**: each entry defines one marker:

- `id`: the marker index within the dictionary
- `size_m`: physical edge length in meters
- `static`: (optional, default false) set to `true` for markers fixed in the scene

**`[[links]]`**: each entry is one measured distance between two markers.
See [Distance Links](#distance-links) for the two kinds.

**`[[mirror_pairs]]`**: each entry declares two markers printed on opposite faces of one rigid board.
See [Mirror Pairs](#mirror-pairs).

### Dictionary IDs

Set `dictionary` to the integer for the ArUco dictionary you printed from.
The common values:

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

The name encodes the marker grid and the count: `DICT_4X4_100` is a 4×4-bit pattern with 100 distinct markers.
A smaller count is more robust to misreads, so pick the smallest dictionary that holds all your marker IDs.

### Corner Numbering

Each marker has four corners, numbered 0 through 3.
Hold a printed marker so its pattern is upright — the way it looks in the GUI preview, with the `ID:` label reading left to right along the bottom.
Then:

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

The saved marker PNGs (from "Save All PNGs") print each corner's index next to it, so you can read the numbers straight off the paper.
Corner 0 is the top-left corner of the upright marker.

## Distance Links

Each `[[links]]` entry records one distance you measured between two markers.
Bundle adjustment uses whatever rigidity you give it.
More links pin the geometry more firmly, but even a single link adds scale information.

A link may only connect two markers of the same kind — both mobile or both static.
The solver skips a mixed static/mobile link, so Caliscope rejects it when the file loads rather than let it silently do nothing.

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

- `marker_a`, `marker_b`: the two marker IDs
- `corner_a`, `corner_b`: the corner index (0–3) on each marker (see Corner Numbering)
- `distance_m`: the measured distance in meters between those two corners

A corner link constrains exactly the one distance you measured, corner to corner.
Use corner links for a rigid fixture you can reach with calipers: two markers mounted on the same board or bracket, close enough that you can touch a caliper jaw to each corner.
Add as many corner links as you have measurements.
Each is independent — there is no requirement that the markers be the same size or face the same way.

**Center links** omit the corner keys, so the distance is measured center to center:

```toml
[[links]]
marker_a = 5
marker_b = 6
distance_m = 0.500
```

- `marker_a`, `marker_b`: the two marker IDs
- `distance_m`: the measured distance in meters between the two marker centers
- (no `corner_a` / `corner_b`)

A marker's center is the middle of its printed square.
Use center links for distances too long for calipers: markers taped to a wall or floor across the room, measured with a tape.
You sight or measure center to center because a tape cannot reliably hit a specific corner across that distance.

A center link fires only in frames where all four corners of both markers are triangulated, since the center is computed from the four corners.

### The Scale-Anchor Case

A common setup is markers fixed to a wall or floor, far apart, with one tape-measure run between two of them recorded as a center link.
This is worth doing, but understand what it buys you.

The per-marker geometry (each marker's own corner spacing) already carries most of the scale information in the reconstruction.
The long tape run does not out-vote it.
What the tape run adds is an **independent** scale check across a long baseline.
If every marker was printed slightly too large or too small — a systematic error from your printer or PDF scaling — every per-marker distance is biased by the same factor, and nothing in the per-marker geometry can notice, because they all agree with each other.
A single tape run measured with a real tape does not share that printing error, so it catches the bias the markers hide.

Think of the center link as a second, independent witness to scale, not as the dominant source of it.

### Measurement Uncertainty: `sigma_m`

Every link carries an assumed uncertainty — how much give-or-take is in your measurement.
That uncertainty is `sigma_m`, in meters.
It is optional.
When you leave it out, Caliscope uses a sensible default for the kind of link:

- **2 mm** (`sigma_m = 0.002`) for corner links — caliper class
- **5 mm** (`sigma_m = 0.005`) for center links — tape-measure class

**Omitting `sigma_m` is the right choice for most users.** The defaults match how well most people measure with the corresponding tool.

Here is what the number does.
During bundle adjustment the solver tries to honor every link, but links can disagree with each other and with the camera views.
`sigma_m` is how it decides who to trust.
A small sigma tells the solver "this distance is nearly exact — bend the reconstruction to match it."
A large sigma tells it "this is a rough figure — match it only if it costs nothing."
So a link with a small sigma pulls harder on the final geometry than a link with a large one.

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

**What not to do:** do not set `sigma_m` smaller than you can actually measure.
An overconfident link — a hand measurement claiming half-millimeter certainty — forces the solver to honor a number that is really wrong.
It cannot bend the distance, so it bends the cameras instead, and your calibration gets worse.
Set `sigma_m` to your honest measurement error, not your hopes.

## Mirror Pairs

Two cameras facing each other cannot both see the same face of a marker.
A mirror pair tells Caliscope that two same-size ArUco markers, printed on opposite faces of one rigid board, share a physical location, so cameras on opposite sides of the board contribute to the same calibration.
This is what the ChArUco board's built-in mirror detection already does for a chessboard pattern, but for arbitrary ArUco markers.
(An earlier `mirror_flag_search` option flipped the image and re-detected, on the assumption that the flipped pattern was the same marker seen from behind. That assumption is unsafe for ArUco — a flipped pattern can decode as a different, valid marker ID. Mirror pairs are the correct replacement.)

A `[[mirror_pairs]]` block records one such pair:

```toml
# Thin laminate: corners merge into shared world points
[[mirror_pairs]]
marker_a = 0
marker_b = 1
anchor_corner_a = 0
anchor_corner_b = 2
thickness_m = 0.0

# Foamboard: corners stay separate, constrained by thickness
[[mirror_pairs]]
marker_a = 5
marker_b = 6
anchor_corner_a = 0
anchor_corner_b = 1
thickness_m = 0.005
# sigma_m = 0.001   # optional; defaults to the 2 mm corner-link class
```

- `marker_a`, `marker_b`: the two marker IDs, one on each face of the board
- `anchor_corner_a`, `anchor_corner_b`: one pair of physically coincident corners (0–3, see Corner Numbering)
- `thickness_m`: the board thickness in meters; 0.0 is allowed
- `sigma_m`: (optional) uncertainty for the thickness constraint; defaults to the 2 mm corner-link class

You only specify one pair of coincident corners.
Caliscope derives the other three from the winding-reversal rule: flipping a square board reverses the winding direction of its corners, so the rest of the mapping follows automatically once one pair is known.
To find your anchor pair, hold the board up to a light so both markers' corners line up, pick any corner of marker A, and read which corner of marker B sits directly behind it.
Write those two numbers down as `anchor_corner_a` and `anchor_corner_b`.

Both markers in a mirror pair must have the same `size_m` — Caliscope validates this.
The anchor derivation assumes both markers' corners sit the same distance from their centers, so a size mismatch produces a wrong mapping.
Both markers must also be both-static or both-mobile, the same rule that governs `[[links]]`.

**Choosing `thickness_m`.** `thickness_m = 0.0` treats corresponding corners on both faces as the same point.
The two faces merge into shared world points, which improves coverage for cameras facing the board from opposite sides and works with as few as two cameras.
A nonzero `thickness_m` keeps each face's own identity and constrains corresponding corners to sit that distance apart — more physically accurate for a substrate with real thickness, but each face must now be independently triangulated, which needs more cameras: roughly two per face.
Use 0.0 for a thin substrate (paper, laminate).
Use the measured thickness for a thick substrate (foamboard, plywood).
There is no fixed cutoff between the two; it is your judgment call based on how thick the board actually is.

**Center alignment.** Both faces must be printed with their centers aligned on the substrate.
A small offset is tolerable: on a 5 mm board with a 2 mm center-to-center offset, the true corner-to-corner distance is √(5² + 2²) = 5.39 mm, a 0.39 mm bias — well within the default 2 mm sigma.
Alignment errors of a few millimeters are tolerable relative to the constraint noise.
Center-align to within the accuracy you want out of the calibration.

**Camera coverage for thick boards.** The four corner-to-corner constraints fix the normal separation between the faces and the two tilt angles, since tilting the board changes the four corner distances asymmetrically.
They cannot fix the two in-plane slides or the in-plane twist between the faces — no network of corner distances can, because small lateral shifts and twists leave all four distances unchanged.
Camera observations resolve slide and twist instead, so a thick-board mirror pair needs solid camera coverage on both faces: three or more cameras per face, or two with a wide baseline.
With barely enough cameras, convergence can be slow.

Zero-thickness mirror pairs have their own two-camera hazard.
A single planar marker seen by two facing cameras sits close to the classic two-fold planar-pose ambiguity when the board is held fronto-parallel to the cameras.
Sweep the board through a range of tilts while capturing; do not hold it flat and square to the cameras.

## Workflow

1. Create or edit `aruco_marker_set.toml` in the workspace targets directory
2. Click "Save All PNGs" in the GUI to generate printable markers with corner labels (saved to `marker_images/` subfolder)
3. Print the markers.
   Mount on rigid backing.
4. Measure distances between markers and record them as `[[links]]` — corner links for caliper-range measurements, center links for tape-measure runs.
   Set `sigma_m` only where your measurement differs from the default tool class.
5. Place static markers in the scene if using any.
   Record calibration video with markers visible across cameras.
   If the solve will recover or refine intrinsics, sweep a mobile marker through depth — toward and away from the cameras — not just side to side.
6. Run extraction, then calibrate.
   Links enter bundle adjustment automatically as rigidity constraints.

## Scripting API

```python
from caliscope.api import calibrate_extrinsics
from caliscope.core.aruco_marker import ArucoMarkerSet
from caliscope.core.constraints import ConstraintSet

marker_set = ArucoMarkerSet.from_toml(toml_path)
constraints = ConstraintSet.from_marker_set(marker_set)

result = calibrate_extrinsics(image_points, cameras, constraints)
result.capture_volume.save("capture_volume")
```

See the [Scripting API](scripting.md) page for the full walkthrough, including extraction and the extrinsic-only variant.
