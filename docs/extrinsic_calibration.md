# Capture Volume: Extrinsic Calibration

<video  controls>
  <source src="../videos/multicamera_calibration.mp4" type="video/mp4">
</video>


## Processing steps

1. Save videos to `project_root/calibration/extrinsic/` according to the naming convention outlined in [Project Setup](project_setup.md#stage-2-extrinsic-calibration)
2. Ensure that videos were synchronized when recording, or provide a [`frame_time_history.csv`](project_setup.md#frame_time_historycsv) file so that caliscope can perform the synchronization during processing.
3. You may need to reload the workspace for the `Calibrate Capture Volume` button to become enabled
4. Pressing `Calibrate Capture Volume` will initiate the calibration. The final log statement at complete will indicate that `point_esimates.toml` has been saved. at this point you can reload the workspace and the `Capture Volume` tab will become enabled.
5. On the `Capture Volume` tab you can visually inspect the relative position of the cameras according to the calibration
6. Set the board origin to a given frame to align the world frame of reference with the board position. This can be refined by flipping the axes.

## Practical Recording Guidelines

1. Ensure Coverage and Overlap:
    - Cover the entire volume where the cameras' fields of view overlap with the Charuco board movements.
    - Ensure there's sufficient overlap in the fields of view of the different cameras. **This overlap is critical for multi-camera calibration.**
  
2. Use a board with sufficiently large squares
    - Larger ArUco markers can be identified from farther away allowing a larger capture volume to be calibrated.

3. Minimize motion blur
    - motion blur can substantially compromise corner recognition
    - using a higher frame rate can reduce motion blur
      - this will require more light to maintain good illumination

4. Consistent Focus:
    - Use manual focus if available to keep the focus consistent throughout the filming. 
    - Auto-focus can introduce inconsistencies 

5. Use the board to define the origin
    - this is for convenience and not a strict requirement 
    - touch the board to the ground while it is held vertically
    - ensure that the top left corner of the board (as shown on the [`Charuco`](calibration_board.md) tab) is in view of the camera and touching the ground