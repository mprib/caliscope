Camera calibration is performed with the aid of a [ChArUco calibration board](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html). This is basically a chessboard that has ArUco markers within it and is used to determine both the intrinsic properties of a camera (optical center, focal length, and lens distortion) as well as where the cameras are relative to each other. 

It is possible to purchase pre-printed calibration boards and then set the parameters of the ChArUco to fit the board you have. **Please keep in mind that row count and column count are not interchangeable, so visually inspect your physical board compared to the board shown in the GUI to make sure they are the same. You may need to swap row and column values.**

<video  controls>
  <source src="../videos/charuco_demo.mp4" type="video/mp4">
</video>

---

## ChArUco Tab Options

ChArUco board creation allows the following configuration options:

### 1. Row and Column Count

While more rows and columns mean that you can get more recognized corners per board view, it results in smaller ArUco markers for a given board size. These smaller markers will then be harder to recognize from a distance, limiting the size of the capture volume you can calibrate for a given resolution of cameras. This presents a trade-off to consider for your set up.

As noted above: row count and column count are not interchangeable. An 8x5 board is not the same as a 5x8 board.
### 2. Board Size

This is the target size that the final printed board will have. This can minimize white space in the `png`. 

### 3. Board Inversion

To save on printer ink, you can select to invert the board image so that white and black regions are swapped. If this is done, then caliscope will invert a grayscale image prior to running ChArUco recognition so that things will still work out.

### 4. Save Calibration Board `.png`

Only used for printing out. This file does not need to be saved anywhere in particular. 

### 5. Save Mirror Board `.png`

In addition to a regular one-sided board, it is possible to print a double-sided board that has the mirror image printed on the back. If the tracker does not find the primary board in a frame, it will then flip the image and search for the board again, flipping the coordinates of any tracked points back to an unflipped frame of reference.

This allows cameras that do not share a common view of a board to better estimate their position relative to each other. The intention is to allow better calibration of systems with surround camera setups as is common in larger scale motion capture. Please note that thicker foam boards may not work well for this. The feature has only been tested with a paper print pressed between two panes of glass.

### 6. Actual Printed Square Size

To ensure that the scale of the world is accurate in your final triangulated points, measure the actual length of a printed square of the board. While errors in this measurement will not cause failure along the way, it can result in very large or very small subjects in the triangulated output. Please note that even if you print directly to your intended board size, small differences in actual square size are likely to result.

---
## Implementation Details


### Taping together a board from multiple printed sheets

Rather than paying for a professionally printed board, it is possible to print a board on multiple standard sheets of paper, trim them as appropriate, then tape them together. Undoubtedly, this will lead to larger errors in calibration, though I have been pleasantly surprised by the quality of the tracked landmark data. 

### Flatness Matters Particularly for Intrinsics

Having a truly flat board is crucial for a good intrinsic camera calibration. At the core of the calibration algorithm is the fact that all points exist on a common plane. Loosely taped pieces of paper or warped cardboard backings will undermine the calibration quality. 

### Different boards for intrinsic and extrinsic calibrations

It is possible to perform the intrinsic calibration and extrinsic calibration using different boards, though you will have to adjust the board definition prior to running each calibration. 

In this way, a more perfectly flat and and aligned board could be used for intrinsic calibration when it is fine to have the board relatively close to a single camera. During extrinsic calibration, a larger board pieced together from multiple sheets of paper could be used to allow calibration of a larger capture volume.

Please note that for a given set of cameras, the intrinsic calibration only needs to be performed once, and that configuration can be copied over to future projects with the same cameras.