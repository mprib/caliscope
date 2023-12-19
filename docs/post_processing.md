*demonstration video coming soon...*



## Processing steps

1. Save videos to `project_root/calibration/extrinsic/` according to the naming convention outlined in [Project Setup](project_setup.md#stage-2-extrinsic-calibration)
2. Ensure that videos were synchronized when recording, or provide a [`frame_time_history.csv`](project_setup.md#frame_time_historycsv) file so that pyxy3d can perform the synchronization during processing.
3. You may need to reload the workspace for the `Calibrate Capture Volume` button to become enabled
4. Pressing `Calibrate Capture Volume` will initiate the calibration. When it is complete, the `Capture Volume` tab will become enabled.

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