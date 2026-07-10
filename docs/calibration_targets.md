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
The embedded markers give the detector a coarse position lock before subpixel corner refinement, which makes it robust to partial occlusion and motion blur.

### Configuration

**Row and Column Count**: Number of squares in each dimension.

!!! warning "Row and column count are not interchangeable"
    An 8x5 board is not a 5x8 board. Compare to the GUI preview if you purchased a pre-printed board.

**Board Size**: Target dimensions for the printed PNG. Does not affect calibration.

**ArUco Scale**: Ratio of ArUco marker size to square size (default 0.75). Editable in the TOML files but not exposed in the GUI.

**Board Inversion**: Swaps black and white to save ink. Caliscope inverts the image during detection when enabled.

**Mirror Board**: Generates a flipped PNG for the back of a double-sided board.
Cameras viewing opposite sides can calibrate against each other.

Caliscope automatically builds rigidity constraints from the board's known corner geometry. There is nothing to configure.

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

A standard checkerboard pattern. Simpler to print than a ChArUco board, but the detector has no ArUco markers for coarse lock, so it is more sensitive to motion blur.

**Row and Column Count**: Same convention as ChArUco boards. Values are not interchangeable.

The GUI offers chessboard for intrinsic calibration only.
The pipeline supports extrinsic use through the [scripting API](scripting.md).

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
