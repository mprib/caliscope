# Cameras: Intrinsic Calibration

<video  controls>
  <source src="../videos/intrinsic_calibration_demo.mp4" type="video/mp4">
</video>

## Processing steps

1. Save calibration video to `project_root/calibration/intrinsic/` with the filename in the format of `port_#.mp4` as described in [Project Setup](project_setup.md#stage-1-intrinsic-calibration)
2. Reload the workspace if needed so that the`Camera` tab becomes enabled 
3. On the specific Camera sub-tab, ensure that the video is loaded correctly, then press `Autocalibrate`
4. The video will play and snapshots of the calibration board will be periodically stored. At the conclusion of the video the calibration will be performed and the updated camera parameters will be displayed in the GUI (and stored in the `config.toml` file at the project root).

**NOTE: Intrinsic calibration only needs to be performed once per camera. Previously determined values can be carried over to a new project's `config.toml` file when using the same cameras in a new setup.** 

## Practical Recording Guidelines
1. Feel free to move the camera *or* the board
    - it can be easier to collect good data when directly monitoring the view of the camera
    - the camera does not need to be in the same position as it is during the extrinsic calibration ([and the calibration board doesn't need to be the same either](calibration_board.md#different-boards-from-intrinsic-and-extrinsic-calibration))

2. Minimize Motion Blur:
    - make movements slow and smooth 
    - Use a high shutter speed to reduce motion blur. 
    - Ensure adequate lighting to allow for a faster shutter speed without underexposing the video.

3. Provide Foreshortening:
    - Hold the calibration board at various angles relative to the camera. This introduces foreshortening, which is crucial for the calibration process as it provides more information about the camera’s lens characteristics.
    - Include a mix of positions: some shots with the board tilted towards the camera, some away, and others at an angle.

4. Cover the Entire Field of View:
    - Move the calibration board throughout the entire field of view of the camera. This ensures that the calibration accounts for lens distortions and other characteristics across the whole image sensor.

5. Use a High-Quality Calibration Board:
    - The board should be printed on a flat, rigid material to prevent warping.

6. Vary the Distance:
    - Film the calibration board at different distances from the camera. This variation helps in understanding how the camera focuses at different depths.

7. Consistent Focus:
    - Use manual focus if available to keep the focus consistent throughout the filming. 
    - Auto-focus can introduce inconsistencies as it may change between shots.

8. Adequate Lighting:
    - Ensure the scene is well-lit to avoid noise and grain in the video, which can interfere with the calibration process.
    - Avoid strong direct light sources that can cause glare or shadows on the calibration board.