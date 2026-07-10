# Cameras: Intrinsic Calibration

Intrinsic calibration determines the internal optical properties of each camera: the focal length (in pixels), the optical center (where the camera's optical axis intersects the sensor), and the lens distortion coefficients that correct barrel and pincushion distortion. These parameters are unique to each camera and remain constant as long as the camera's focal length and lens haven't changed.

!!! note "This step is optional for many setups"
    Extrinsic calibration can recover focal length and leading distortion during bundle adjustment, if you capture for it. Fisheye cameras still require this step, and it remains the way to get a dense lens model. See [Skipping Intrinsic Calibration](extrinsic_calibration.md#skipping-intrinsic-calibration) for the prerequisites.

### Calibration Targets

Caliscope supports both **ChArUco** and **chessboard** patterns for intrinsic calibration. ChArUco boards tolerate partial occlusion and provide unique corner identification, so they tend to produce better results. See [Calibration Targets](calibration_targets.md) for details on creating your calibration board.

Physical board size does not affect intrinsic results.
See [Calibration Targets](calibration_targets.md#intrinsic-calibration-size-independent) for why.

## Processing Steps

1. Place calibration videos in `calibration/intrinsic/` with filenames in the format `cam_N.mp4` (e.g., `cam_0.mp4`, `cam_1.mp4`) as described in [Project Setup](project_setup.md#stage-1-intrinsic-calibration)
2. The Cameras tab will enable automatically when videos are detected
3. On the specific camera sub-tab, confirm that the video loaded correctly
4. Confirm by scrolling through the video that the calibration board corners are being recognized (red dots placed on them)
5. **Option 1: Manual Board Selection**
   1. Scroll through the calibration footage and select `Add Grid` to include the frame in your calibration data. Grid images should accumulate for all grids included in the intrinsic calibration.
   2. When you have chosen the frames you like, click `Calibrate` to begin the calibration process.
6. **Option 2: Autocalibrate**
   1. Select the target number of boards for your calibration (~20 works well)
   2. Select the percent of the board that must be identified for it to be included in the calibration data (the "Board Threshold")
   3. Click `Autocalibrate`
   4. The video will play and calibration data will be periodically stored. When the video finishes, calibration runs and the updated camera parameters appear in the GUI.

## Reusability

**Intrinsic calibration only needs to be performed once per camera.** The same calibration parameters can be used across multiple projects as long as:

- The camera's focal length hasn't changed (no zoom adjustment)
- The lens hasn't been physically modified or replaced
- You're using the same focus setting (see warning below)

You can copy previously determined intrinsic parameters from one project to another when reusing the same cameras in a new setup. Intrinsic parameters are stored in `camera_array.toml` (TOML format) within the project's calibration directory.

### Camera Model Considerations

Caliscope supports two distortion models:

- **Standard (Brown-Conrady)**: Pinhole model with 5 distortion coefficients (k1, k2, p1, p2, k3) for radial and tangential distortion. Works well for most cameras, including typical webcams and professional cameras with moderate field-of-view lenses.
- **Fisheye (equidistant)**: 4-coefficient model (k1, k2, k3, k4) for extreme wide-angle lenses like GoPros and action cameras. Enable this by setting the `fisheye` flag in `camera_array.toml` for the relevant camera.

If your calibration results show poor reprojection accuracy with a wide-angle lens, try enabling the fisheye model for that camera.

## Recording Tips

- Move the camera, the board, or both. You do not need the same board or camera position as extrinsic calibration.
- Tilt the board at various angles (foreshortening gives the solver more lens geometry information).
- Cover the entire field of view and vary the distance from the camera.
- Move slowly. Use manual focus and good lighting to avoid motion blur and noise.
- Print the board on flat, rigid material. Warping introduces systematic error.

## Programmatic Workflow

Intrinsic calibration can be run from a Python script without the GUI. See the [Scripting API](scripting.md#step-3-intrinsic-calibration) page for a step-by-step walkthrough.
