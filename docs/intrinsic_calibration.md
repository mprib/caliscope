# Cameras: Intrinsic Calibration

<video  controls>
  <source src="../videos/intrinsic_calibration_demo.mp4" type="video/mp4">
</video>

## Overview

Intrinsic calibration determines the internal optical properties of each camera:

- **Focal length** (in pixels) — how the camera magnifies the scene
- **Optical center** — the pixel coordinates where the camera's optical axis intersects the sensor
- **Lens distortion coefficients** — correction parameters for barrel/pincushion distortion

These parameters are unique to each camera and remain constant as long as the camera's focal length and lens haven't changed.

### Calibration Targets

Caliscope supports both **ChArUco** and **chessboard** patterns for intrinsic calibration. ChArUco boards are generally preferred because they are more robust to partial occlusion and provide unique corner identification. See [Calibration Targets](calibration_targets.md) for details on creating your calibration board.

### Physical Board Size

The physical dimensions of your calibration board do not affect the final intrinsic parameters (focal length and distortion coefficients), which are scale-invariant. Whether your board is 10 cm or 1 meter across, these parameters are identical. The board dimensions do affect internal pose estimates during optimization, but approximate dimensions are acceptable. This is why you can use different calibration boards for intrinsic and extrinsic calibration without issue.

## Processing Steps

1. Place calibration videos in `calibration/intrinsic/` with filenames in the format `cam_N.mp4` (e.g., `cam_0.mp4`, `cam_1.mp4`) as described in [Project Setup](project_setup.md#stage-1-intrinsic-calibration)
2. The Cameras tab will enable automatically when videos are detected
3. On the specific camera sub-tab, ensure that the video loaded correctly
4. Confirm by scrolling through the video that the calibration board corners are being recognized (red dots placed on them)
5. **Option 1: Manual Board Selection**
   1. Scroll through the calibration footage and select `Add Grid` to include the frame in your calibration data. Grid images should accumulate for all grids included in the intrinsic calibration.
   2. When you have chosen the frames you like, click `Calibrate` to begin the calibration process.
6. **Option 2: Autocalibrate**
   1. Select the target number of boards for your calibration (~20 works well)
   2. Select the percent of the board that must be identified for it to be included in the calibration data (the "Board Threshold")
   3. Click `Autocalibrate`
   4. The video will play and calibration data will be periodically stored. At the conclusion of the video, calibration will be performed and the updated camera parameters will be displayed in the GUI.

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

## Practical Recording Guidelines

### Camera and Board Movement
- Feel free to move the camera *or* the board
  - It can be easier to collect good data when directly monitoring the view of the camera
  - The camera does not need to be in the same position as it is during the extrinsic calibration (and the calibration board doesn't need to be the same either)

### Minimize Motion Blur
- Make movements slow and smooth
- Use a high shutter speed to reduce motion blur
- Ensure adequate lighting to allow for a faster shutter speed without underexposing the video

### Provide Foreshortening
- Hold the calibration board at various angles relative to the camera. This introduces foreshortening, which is crucial for the calibration process as it provides more information about the camera's lens characteristics.
- Include a mix of positions: some shots with the board tilted towards the camera, some away, and others at an angle

### Cover the Entire Field of View
- Move the calibration board throughout the entire field of view of the camera. This ensures that the calibration accounts for lens distortions and other characteristics across the whole image sensor.

### Use a High-Quality Calibration Board
- The board should be printed on a flat, rigid material to prevent warping
- Even slight bowing or warping will introduce systematic errors into your calibration

### Vary the Distance
- Film the calibration board at different distances from the camera. This variation helps in understanding how the camera focuses at different depths.

### Consistent Focus Settings
- **Use manual focus if available** to keep the focus consistent throughout the filming
- **WARNING**: If your camera uses auto-focus, the focal length changes between shots, which invalidates the calibration. Auto-focus can introduce inconsistencies as the focus mechanism changes between frames. If you cannot disable auto-focus, ensure the camera is focused at a fixed distance and doesn't refocus during recording.

### Adequate Lighting
- Ensure the scene is well-lit to avoid noise and grain in the video, which can interfere with the calibration process
- Avoid strong direct light sources that can cause glare or shadows on the calibration board
