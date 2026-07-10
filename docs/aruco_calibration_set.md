# The ArUco Calibration Set

Extrinsic calibration with ArUco markers is configured through a TOML file: `calibration/targets/aruco_marker_set.toml`.
The file lists which markers exist, how large they are, which are fixed in the scene, and what distances you measured between them.

If you are deciding whether to use ArUco markers rather than a ChArUco board, start with [Calibration Targets](calibration_targets.md).

## The TOML File

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

[[links]]
marker_a = 0
corner_a = 1
marker_b = 1
corner_b = 3
distance_m = 0.204

[[links]]
marker_a = 5
marker_b = 6
distance_m = 0.500
```

**`dictionary`**: the OpenCV ArUco dictionary as an integer (see [Dictionary IDs](#dictionary-ids)).
Pick the smallest dictionary that holds all your marker IDs.

**`[[markers]]`**: one entry per marker.

- `id`: marker index within the dictionary
- `size_m`: physical edge length in meters. Measure with calipers, not from the PDF. This value also sets world scale.
- `static` (optional, default false): `true` for markers fixed in the scene

**`[[links]]`** (optional): measured distances between markers. See [Distance Links](#distance-links).

**`[[mirror_pairs]]`** (optional): markers printed on opposite faces of one board. See [Mirror Pairs](#mirror-pairs).

## Static and Mobile Markers

Markers are **mobile** by default (carried through the volume) or **static** (`static = true`, fixed in the scene).

Static markers anchor the reconstruction: Caliscope treats their corners as shared 3D points across all frames.
Mobile markers provide coverage. If you skip intrinsic calibration or refine intrinsics, sweep a mobile marker toward and away from the cameras to provide the depth variation the solver needs (see [the depth-ratio gate](extrinsic_calibration_reference.md#the-depth-ratio-gate)).

Two rules:

1. Links and mirror pairs must join same-class markers (static-static or mobile-mobile). Caliscope rejects a mixed link at load time.
2. A static marker that moved during capture is detected and dropped before optimization, with a log warning.

## Editing in the GUI

The TOML file is the editing interface. The GUI shows a read-only summary.

Select **ArUco** as the extrinsic target on the Project tab. Three buttons:

- **Edit TOML**: opens the file in your text editor
- **Reload**: re-reads after you edit
- **Save All PNGs**: writes printable markers with corner labels to `marker_images/`

## Distance Links

Each `[[links]]` entry is one distance you measured between two markers.
More links pin the geometry more firmly, but even one adds scale information.

**Corner links** name a specific corner on each marker:

```toml
[[links]]
marker_a = 0
corner_a = 1
marker_b = 1
corner_b = 3
distance_m = 0.204
```

**Center links** omit the corner keys and measure center-to-center:

```toml
[[links]]
marker_a = 5
marker_b = 6
distance_m = 0.500
```

A center link between two fixed markers catches systematic printing error that per-marker geometry cannot detect, since all markers from the same printer share the same bias.

### Corner Numbering

```
  0 ──── 1
  │      │
  │      │
  3 ──── 2
```

Hold the marker upright as it appears in the GUI preview. The saved PNGs label each corner.

### Measurement Uncertainty: `sigma_m`

Links carry an optional `sigma_m` (meters). Defaults: 2 mm for corner links, 5 mm for center links. Omitting it is usually right.

Override when your measurement is meaningfully better or worse:

| Method | `sigma_m` |
|--------|-----------|
| Machined/3D-printed fixture | `0.0005` |
| Long tape with sag | `0.01` |

`sigma_m` should reflect your actual measurement accuracy. A value tighter than your real precision will pull the geometry toward a wrong distance.

## Mirror Pairs

A mirror pair declares two same-size markers printed on opposite faces of one rigid board.

```toml
[[mirror_pairs]]
marker_a = 0
marker_b = 1
anchor_corner_a = 0
anchor_corner_b = 2
thickness_m = 0.0
```

- `anchor_corner_a`, `anchor_corner_b`: one pair of physically coincident corners. Hold the board to a light and read which corner of marker B sits behind a corner of marker A.
- Both markers must have the same `size_m` and be both-static or both-mobile.

- `thickness_m = 0.0` (paper, laminate): marker B's corners collapse into marker A's. Both sides share four 3D points, so cameras facing opposite sides of the board link through shared points. This is what makes mirror boards useful for surround setups where opposing cameras never share a direct view.
- Nonzero (foamboard, plywood): all eight corners stay separate, connected by distance constraints at the measured thickness.

## Dictionary IDs

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

The name encodes the grid and count: `DICT_4X4_100` is a 4x4-bit pattern with 100 distinct markers.

## Workflow

1. Create or edit `aruco_marker_set.toml`
2. Save All PNGs from the GUI
3. Print and mount on rigid backing
4. Measure distances between markers, record as `[[links]]`
5. Place static markers in the scene. Record calibration video.
6. Run extraction, then calibrate. Links enter bundle adjustment automatically.

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

See the [Scripting API](scripting.md) page for the full walkthrough.
