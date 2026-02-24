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

<video controls>
  <source src="../videos/charuco_demo.mp4" type="video/mp4">
</video>

### Configuration Parameters

**Row and Column Count**: The number of squares in each dimension. **These values are NOT interchangeable.** An 8x5 board is not the same as a 5x8 board. If you purchase a pre-printed board, visually compare it to the preview in the GUI to ensure row and column values match the physical board.

More rows and columns provide more corners per view, improving calibration quality. However, they also result in smaller ArUco markers for a given board size. Smaller markers are harder to detect from a distance, limiting the maximum size of your capture volume. Consider this trade-off when choosing dimensions for your setup.

**Board Size**: The target dimensions for the printed image. This parameter controls the PNG output size and minimizes white space around the board. It does not affect calibration quality.

**ArUco Scale**: The ratio of the ArUco marker size to the chessboard square size within the board. The default is 0.75. Increasing this value makes the markers larger (easier to detect at distance) but reduces the white border around each marker, which can undermine recognition if set too high. This parameter is not exposed in the GUI but can be edited in `calibration/targets/intrinsic_charuco.toml` or `extrinsic_charuco.toml`.

**Board Inversion**: To save printer ink, you can invert the board so black and white regions are swapped. Caliscope automatically inverts the grayscale camera image during detection when this option is enabled, so recognition still works correctly.

**Mirror Board**: For surround camera setups, the mirror board PNG contains the pattern flipped horizontally for printing on the back of the board. When the tracker does not find the primary board in a frame, it automatically flips the image and searches again, then maps any detected points back to the standard coordinate frame.

This allows cameras viewing opposite sides of the board to calibrate against each other, which is necessary for systems with cameras arranged around the capture volume. The feature has been tested with paper prints pressed between two panes of glass. Thicker foam boards may not work reliably due to alignment issues between front and back.

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

### Advantages

- **Detection range**: Single markers can be printed at poster size or larger, making them visible from much greater distances than board-based targets.
- **Spatial coverage**: Multiple markers can be scattered around a laboratory or capture volume.
- **Double-sided printing**: Like mirror boards, ArUco markers can be printed front and back on rigid material for surround setups.

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

### Flatness Requirements

**Flatness is critical for intrinsic calibration.** The calibration algorithm assumes all detected points lie on a perfect plane. Warped cardboard, loosely taped paper, or bowed foam board will introduce systematic errors that undermine calibration quality. Use rigid backing and ensure the surface is truly flat.

**Flatness is less critical for extrinsic calibration.** Bundle adjustment can absorb small deviations from planarity because it optimizes camera poses across many frames. However, excessive warping will still degrade results.

### Measure Actual Printed Dimensions

Even when printing at exact scale, small variations in printed size are common. Measure the actual dimensions of your printed target with a ruler or calipers before entering values in the GUI. For both ChArUco boards and ArUco markers, measure corner to corner. Enter this value in the "Printed Edge" field.

### Physical Construction

- **Rigid backing**: Tape or mount your printed target to cardboard, foam board, or glass. For mirror boards, two panes of glass with the print between them work well.
- **Avoid glare**: Matte finishes reduce specular reflections that can interfere with corner detection. If using glass, consider placing the print between two panes rather than on the surface.
- **Alignment**: For double-sided boards, ensure front and back prints are precisely aligned. Misalignment will cause tracking failures when the mirrored view is used.

### Different Boards for Different Stages

You can use different calibration targets for intrinsic and extrinsic calibration, and this is often the better approach. A small, perfectly flat board works well for intrinsic calibration when the target can be held close to the camera. A larger board or multiple ArUco markers may be necessary for extrinsic calibration of a large capture volume.

Remember that **intrinsic calibration only needs to be performed once per camera**. Once you have calibrated your lenses, you can copy the intrinsic parameters to new projects and skip directly to extrinsic calibration.
