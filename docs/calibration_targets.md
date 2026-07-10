# Calibration Targets

Camera calibration in Caliscope requires a physical target with known geometry.
The target serves two purposes: determining the intrinsic properties of each camera (optical center, focal length, lens distortion), and establishing the spatial relationship between cameras (extrinsic calibration).

Caliscope supports three types of calibration targets:

| Target | Intrinsic | Extrinsic | Provides Scale |
|--------|-----------|-----------|----------------|
| ChArUco Board | Yes | Yes | Yes |
| Chessboard | Yes | No | No (intrinsic only) |
| ArUco Markers | No | Yes | Yes |

---

## ChArUco Board

The [ChArUco board](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html) is the default and recommended calibration target.
It combines a chessboard pattern with embedded ArUco markers, enabling both intrinsic and extrinsic calibration from the same target.


### Configuration Parameters

**Row and Column Count**: The number of squares in each dimension.
**These values are NOT interchangeable.** An 8x5 board is not the same as a 5x8 board.
If you purchase a pre-printed board, compare it to the preview in the GUI to verify.

**Board Size**: The target dimensions for the printed image.
Controls the PNG output size.
Does not affect calibration quality.

**ArUco Scale**: The ratio of ArUco marker size to square size (default 0.75).
Not exposed in the GUI but editable in `calibration/targets/intrinsic_charuco.toml` or `extrinsic_charuco.toml`.

**Board Inversion**: Swaps black and white to save ink.
Caliscope automatically inverts the camera image during detection when enabled.

**Mirror Board**: For surround camera setups, generates a horizontally flipped PNG for printing on the back of the board.
When the primary pattern is not found in a frame, Caliscope flips the image and searches again, mapping detected points back to the standard coordinate frame.
This allows cameras viewing opposite sides of the board to calibrate against each other.

### Save Options

- **Save Calibration Board PNG**: Generate the primary board image for printing.
  This file does not need to be stored in any particular location.
- **Save Mirror Board PNG**: Generate the flipped board image for double-sided printing.
  Save this separately and print it on the reverse side of the primary board.

### Rigidity Constraints (Automatic)

A ChArUco board is a rigid object with known geometry: every pair of chessboard corners sits a fixed distance apart.
Caliscope now feeds that geometry into bundle adjustment as rigidity constraints.
The optimizer knows the true distances between corners and pulls the reconstruction toward them, which stabilizes the solution and improves scale accuracy.

This happens automatically.
There is nothing to configure and no TOML to edit.
If you calibrate with a ChArUco board, the constraints are built from the board you already defined.

---

## Chessboard

A standard checkerboard pattern provides a simpler alternative for intrinsic calibration.
Chessboards require no specialized printing.

### Configuration Parameters

**Row and Column Count**: The number of squares in each dimension (the same convention as ChArUco boards).
Caliscope derives the internal corner count automatically.
As with ChArUco boards, these values are not interchangeable.

### Limitations

Chessboards **cannot be used for extrinsic calibration**.
Unlike ChArUco boards, which have uniquely identifiable ArUco markers, a chessboard provides no way to track specific points across multiple frames or establish correspondence between different camera views.
Use chessboard targets only when you need to calibrate lens distortion and have a separate method for extrinsic calibration.

---

## ArUco Markers

Caliscope supports extrinsic calibration with one or more ArUco markers.
Markers can be mobile (carried through the volume) or static (fixed in the scene at surveyed positions), and known distances between them provide rigid body constraints that anchor the calibration.

This approach is particularly useful for large capture volumes where a full board would be too small to detect reliably.
Large-format markers printed on a plotter, or several markers surveyed into the scene, stay detectable at distances where a ChArUco board's small squares blur together.

A plain ArUco scene provides far fewer detection points per frame than a ChArUco board, so rigidity constraints hold the solve together.
The baseline constraints are automatic: each marker is a square of known size, and Caliscope derives its corner-to-corner geometry from `size_m`.
Static reference markers and measured inter-marker distances strengthen the solve further.

The marker set (marker IDs and sizes, static flags, distance links, mirror pairs, and measurement uncertainties) is configured in `aruco_marker_set.toml` and summarized in the GUI.
See [The ArUco Calibration Set](aruco_calibration_set.md) for the full configuration guide.

### Limitations

ArUco markers **cannot be used for the intrinsic calibration step**.
A handful of marker corners does not provide the dense image coverage the per-camera solver needs to estimate a full distortion model.
Either calibrate intrinsics with a ChArUco board or chessboard first, or skip the step and let extrinsic calibration recover intrinsics jointly.
See [Skipping Intrinsic Calibration](extrinsic_calibration.md#skipping-intrinsic-calibration) for the prerequisites.

---

## Physical Size and World Scale

Physical target dimensions affect intrinsic and extrinsic calibration differently.

### Intrinsic Calibration: Size-Independent

**Intrinsic calibration does not use physical size.** The lens distortion model depends only on pixel coordinates and their geometric relationships in the image plane.
Whether your calibration target is 10 cm or 1 meter across, the intrinsic parameters (focal length in pixels, distortion coefficients, optical center) remain identical.
The algorithm only needs to see the same pattern from multiple viewing angles.

This means you can measure and enter the physical size of your target after intrinsic calibration is complete, or even use different-sized boards for intrinsic and extrinsic stages.

### Extrinsic Calibration: Physical Size Defines World Scale

**Extrinsic calibration requires physical size.** The physical dimension you measure and enter for your calibration target becomes the **scale gauge** for the entire 3D reconstruction.
All output coordinates are in meters, scaled relative to this measurement.

You measure the target, enter the dimension, and all 3D output is in meters at that scale.
If the measurement is wrong, all coordinates are wrong by the same factor, and nothing in the calibration will flag it.
Use calipers, not a ruler.

---

## Same-as-Intrinsic Option

When using a ChArUco board for both intrinsic and extrinsic calibration, Caliscope provides a "Same as Intrinsic" checkbox that links the two configurations.
Enable this option to avoid entering row count, column count, and square size twice.
If you need different boards for the two stages (e.g., a small flat board for intrinsic calibration and a large assembled board for extrinsic calibration), leave this option disabled.

---

## Printing and Preparation

Mount your printed target on rigid, flat backing (cardboard, foam board, or glass).
The intrinsic solver assumes detected points lie on a perfect plane.
Use a matte finish to reduce glare.
Measure the actual printed dimensions with calipers before entering values in the GUI.
