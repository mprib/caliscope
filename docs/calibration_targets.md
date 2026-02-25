# Calibration Targets

Camera calibration in Caliscope requires a physical target with known geometry. The target serves two purposes: determining the intrinsic properties of each camera (optical center, focal length, lens distortion), and establishing the spatial relationship between cameras (extrinsic calibration).

Caliscope supports three types of calibration targets:

| Target | Intrinsic | Extrinsic | Provides Scale |
|--------|-----------|-----------|----------------|
| ChArUco Board | Yes | Yes | Yes |
| Chessboard | Yes | No | No (intrinsic only) |
| ArUco Marker | No | Yes | Yes |

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

---

## Chessboard

A standard checkerboard pattern provides a simpler alternative for intrinsic calibration. Chessboards require no specialized printing.

### Configuration Parameters

**Row and Column Count**: The number of squares in each dimension (the same convention as ChArUco boards). Caliscope derives the internal corner count automatically. As with ChArUco boards, these values are not interchangeable.

### Limitations

Chessboards **cannot be used for extrinsic calibration**. Unlike ChArUco boards, which have uniquely identifiable ArUco markers, a chessboard provides no way to track specific points across multiple frames or establish correspondence between different camera views. Use chessboard targets only when you need to calibrate lens distortion and have a separate method for extrinsic calibration.

---

## ArUco Marker

A single ArUco marker can be used for extrinsic calibration. This approach is particularly useful for large capture volumes where a full board would be too small to detect reliably.

### Configuration Parameters

**ArUco Dictionary**: The marker set from which the marker ID is drawn. Each dictionary defines a set of markers varying in count and error-correction strength.

**Marker ID**: The specific marker index within the chosen dictionary.

**Marker Size**: The physical side length of the printed marker in centimeters, measured corner to corner. This value sets the world scale for extrinsic calibration.

### Limitations

ArUco markers **cannot be used for intrinsic calibration**. A single marker does not provide enough geometric constraints to estimate lens distortion parameters. Use a ChArUco board or chessboard for intrinsic calibration before proceeding to ArUco-based extrinsic calibration.

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
