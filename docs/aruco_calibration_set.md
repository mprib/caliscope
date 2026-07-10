# The ArUco Calibration Set

Extrinsic calibration with ArUco markers is configured through a **marker set**: which markers exist, how large they are, which are fixed in the scene, and what distances you measured between them.
The set is defined in `aruco_marker_set.toml` in the workspace targets directory (`calibration/targets/`).
If you are deciding *whether* to use ArUco markers rather than a ChArUco board, start with [Calibration Targets](calibration_targets.md).
If you are calibrating without prior intrinsic calibration, the marker set carries extra weight; see [skipping intrinsic calibration](extrinsic_calibration.md#skipping-intrinsic-calibration).

## Rigidity Constraints

ArUco markers give four corners each, far fewer than a ChArUco board's ~70.
Rigidity constraints stabilize the solve when point density is low.

**The baseline constraints are automatic.** Caliscope builds the full set of corner-to-corner distances for every marker from its `size_m`.
You do not configure this.
Measure the printed marker with calipers rather than trusting the PDF, because `size_m` also sets the world scale.

Beyond the baseline:

- **Static markers** anchor the scene geometry.
- **Distance links** tie separate markers together and add an independent check on world scale (see [The Scale-Anchor Case](#the-scale-anchor-case)).
- **Mirror pairs** let cameras on opposite sides of a double-sided board calibrate against each other.

None of these is mandatory, but static anchors plus a mobile marker swept through the volume gives the solver the most to work with.

## Static and Mobile Markers

Every marker in the set is either **mobile** (the default) or **static** (`static = true`).
The distinction runs through everything else on this page.

**Static markers** are fixed in the scene: taped to a wall, bolted to a floor plate.
Caliscope treats a static marker's corners as one set of 3D points shared across every frame, so every observation from every camera and every moment pins the same four points.
This anchors the reconstruction.

**Mobile markers** are carried through the volume.
They do more than fill space: a mobile marker swept *toward and away from* the cameras supplies the depth variation that makes focal length observable.
If you skip intrinsic calibration or let the solver refine intrinsics, this near-to-far motion is what the solver feeds on.
See [the depth-ratio gate](extrinsic_calibration.md#the-depth-ratio-gate) for why moving at a constant distance from the cameras is not enough.

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
A smaller count resists misreads better, so pick the smallest dictionary that holds all your marker IDs.

### Corner Numbering

Each marker has four corners, numbered 0 through 3.
Hold a printed marker so its pattern is upright, the way it looks in the GUI preview, with the `ID:` label reading left to right along the bottom.
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

A link may only connect two markers of the same kind: both mobile or both static.
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
Each is independent.
There is no requirement that the markers be the same size or face the same way.

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

A tape-measure distance between two fixed markers catches systematic printing error that per-marker geometry cannot detect on its own, since all markers from the same printer share the same bias.

### Measurement Uncertainty: `sigma_m`

Every link carries an assumed uncertainty: how much give-or-take is in your measurement.
That uncertainty is `sigma_m`, in meters.
It is optional.
When you leave it out, Caliscope uses a sensible default for the kind of link:

- **2 mm** (`sigma_m = 0.002`) for corner links (caliper class)
- **5 mm** (`sigma_m = 0.005`) for center links (tape-measure class)

**Omitting `sigma_m` is usually the right choice.** The defaults match how well calipers and tape measures typically perform.

A smaller sigma pulls harder on the final geometry.
Override when your measurement is meaningfully better or worse than the default:

| Method | `sigma_m` | Rationale |
|--------|-----------|-----------|
| Machined/3D-printed fixture | `0.0005` | CAD-known distance, sub-mm certainty |
| Long tape with sag | `0.01` | Sag and parallax limit accuracy to ~1 cm |
| Eyeballed hand measurement | `0.02` | Contributes without overriding better data |

Do not set `sigma_m` smaller than you can actually measure.
An overconfident link forces the solver to bend the cameras to match a wrong number.

## Mirror Pairs

A mirror pair tells Caliscope that two same-size markers are printed on opposite faces of one rigid board, so cameras on opposite sides contribute to the same calibration.

```toml
[[mirror_pairs]]
marker_a = 0
marker_b = 1
anchor_corner_a = 0
anchor_corner_b = 2
thickness_m = 0.0       # thin substrate: merge corners into shared points
# thickness_m = 0.005   # thick substrate: constrain corners by board thickness
```

- `anchor_corner_a`, `anchor_corner_b`: one pair of physically coincident corners. Hold the board up to a light and read which corner of marker B sits directly behind a corner of marker A. Caliscope derives the other three.
- `thickness_m = 0.0` for thin substrates (paper, laminate). Use the measured thickness for thick substrates (foamboard, plywood).
- Both markers must have the same `size_m` and be both-static or both-mobile.

Thick boards need more camera coverage per face (roughly two cameras per side).
Align both faces' centers on the substrate.
Sweep the board through a range of tilts while capturing; do not hold it flat to the cameras.

## Workflow

1. Create or edit `aruco_marker_set.toml` in the workspace targets directory
2. Click "Save All PNGs" in the GUI to generate printable markers with corner labels (saved to `marker_images/` subfolder)
3. Print the markers.
   Mount on rigid backing.
4. Measure distances between markers and record them as `[[links]]`.
   Use corner links for caliper-range measurements, center links for tape-measure distances.
   Set `sigma_m` only where your measurement is meaningfully better or worse than the default.
5. Place static markers in the scene if using any.
   Record calibration video with markers visible across cameras.
   If the solve will recover or refine intrinsics, move a mobile marker toward and away from the cameras, not just side to side.
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
