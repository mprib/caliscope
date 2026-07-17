# Calibration Targets

Caliscope requires a physical target with known geometry.
Some targets serve both intrinsic and extrinsic calibration; others serve only one.

| Target | Intrinsic | Extrinsic |
|--------|-----------|-----------|
| ChArUco Board | Yes | Yes |
| ArUco Markers | No | Yes |
| Chessboard | Yes | Scripting only |

---

## ChArUco Board

The [ChArUco board](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html) is the recommended target.
It combines a chessboard pattern with embedded ArUco markers, enabling both intrinsic and extrinsic calibration from the same target.
The markers identify each corner, so a partial view of the board still yields usable points.

### Configuration

**Row and Column Count**: Number of squares in each dimension.

!!! warning "Row and column count are not interchangeable"
    An 8x5 board is not a 5x8 board. Compare to the GUI preview if you purchased a pre-printed board.

**Board Size**: Target dimensions for the printed PNG. Does not affect calibration.

**ArUco Scale**: Ratio of ArUco marker size to square size (default 0.75). Editable in the TOML files but not exposed in the GUI.

**Board Inversion**: Swaps black and white to save ink. Caliscope inverts the image during detection when enabled.

**Mirror Board**: Generates a flipped PNG for the back of a double-sided board.
Cameras viewing opposite sides can calibrate against each other.

Caliscope automatically builds rigidity constraints from the board's known corner geometry.

### Two-Sided Boards and Thickness

A zero-thickness board gives the best results: print the pattern and its mirror on paper, and press them back-to-back under glass or acrylic.
Front and back detections then collapse to shared 3D points, which couples opposing cameras far more rigidly than any distance constraint.
It also halves the camera floor, from four down to two.
If you can build a thin board, do that and leave thickness at zero.

If the board has a real substrate (foam core, gatorboard), set its thickness so the two faces are modeled as separate, rigidly linked point sets.
Thickness is set in the charuco TOML (or via the scripting API):

```toml
# calibration/targets/extrinsic_charuco.toml
thickness_cm = 0.6   # measured substrate thickness
```

There is deliberately no GUI field: the thickness is baked into the landmark data at extraction time, and an accidental edit after extraction makes that data wrong.

When the extrinsic board is set to "same as intrinsic", the extrinsic role reads `intrinsic_charuco.toml`. Set the value there instead.

When using a thick board:

- **Set the measured thickness before extracting landmarks.** If the value changes after extraction, calibration refuses to run until you re-extract (or restore the value).
- **Mounting convention**: mount the mirror print flipped about its **vertical axis**, edges aligned with the front sheet. Each back corner then sits directly behind its front counterpart, which is the geometry the constraints assume.
- **Turn the board so the front-viewing and back-viewing cameras trade places over the session.** Cameras are linked only by seeing the same face at the same instant. If one fixed group of cameras only ever sees the front and another only ever sees the back, the two groups never link: the calibration poses the larger group and leaves the rest unposed. Turning the board through the volume bridges them.
- **A thick board needs at least four cameras, where a zero-thickness board needs two.** In a good share of frames, at least two cameras must see the front face while at least two others see the back. A face seen by a single camera cannot be triangulated, and the thickness constraints only act in frames where both faces have triangulated points. Those frames are what rigidly link the front-viewing and back-viewing cameras. Calibration stops with an error if no such frame exists. Since no camera can see both faces at once, that means two per face. A zero-thickness board escapes this: its faces share the same 3D points, so one camera on each side sees the same points and triangulates them directly.
- **Origin note**: setting the world origin from the board anchors to the front face. The origin plane sits recessed into the board by the thickness when viewed from the mirror side.

Two-sided chessboards are not supported. The chessboard tracker has no mirror detection path. Use a ChArUco board.

---

## ArUco Markers

ArUco markers are useful for large capture volumes where a ChArUco board's small squares blur together at distance.
Large-format prints or multiple markers surveyed into the scene stay detectable across wide spaces.

ArUco markers cannot drive intrinsic calibration.
Intrinsic calibration requires many coplanar points observed from varied angles and distances.
A single marker gives only four corners per view.
Calibrate intrinsics with a ChArUco board or chessboard first.

The marker set is configured in a TOML file.
See [The ArUco Calibration Set](aruco_calibration_set.md) for the full guide.

---

## Chessboard

A standard checkerboard pattern. Simpler to print than a ChArUco board, but the whole board must be visible: OpenCV's chessboard detector is all-or-nothing, so a corner cut off by the frame edge or covered by a hand loses the entire view.

**Row and Column Count**: Same convention as ChArUco boards. Values are not interchangeable.

The GUI offers chessboard for intrinsic calibration only.
The pipeline supports extrinsic use through the [scripting API](scripting.md#chessboard-extrinsics).

!!! warning "Board shape matters for extrinsic use"
    Pick a chessboard with one odd and one even inner-corner count.
    A board of N by M squares has N-1 by M-1 inner corners.
    A board with both counts even, or both odd, looks identical after a half turn.
    Its corner ids can then reverse between cameras whose views differ by that much, which corrupts triangulation.
    Intrinsic calibration is unaffected.
    See [Chessboard extrinsics](scripting.md#chessboard-extrinsics) for details.

---

## Physical Size and World Scale

Intrinsic calibration does not use physical size.
The lens model depends only on pixel geometry.
You can measure your target after intrinsic calibration, or use different-sized boards for the two stages.

!!! info "Physical size defines world scale"
    The dimension you measure and enter for extrinsic calibration becomes the scale gauge for all 3D output. If the measurement is wrong, all coordinates are wrong by the same factor.

---

## Same-as-Intrinsic Option

When using a ChArUco board for both stages, the GUI's "Same as Intrinsic" checkbox copies the board configuration.
Leave it disabled if you need different boards for intrinsic and extrinsic calibration.

---

## Printing and Preparation

Mount on rigid, flat backing (cardboard, foam board, or glass).
Use a matte finish to reduce glare.
Measure the actual printed dimensions with calipers before entering values in the GUI.
