# Post-Processing: Landmark Triangulation from Motion Capture


*demonstration video coming soon...*


## Processing steps

1. Save videos to dedicated subfolders within `project_root/recordings/` according to the naming convention outlined in [Project Setup](project_setup.md#stage-3-processing-motion-capture-trial)
2. Ensure that videos were synchronized when recording, or provide a [`frame_time_history.csv`](project_setup.md#frame_time_historycsv) file so that caliscope can perform the synchronization during processing.
3. You may need to reload the workspace for the recordings to appear in the `PostProcessing` tab
4. Select which tracker you would like to apply
5. Click the `Process` button to begin the landmark tracking and triangulation.
6. 3D landmark positions will be visualized and you can open the subfolder to inspect the landmark tracking on the recordings or to access the trajectory output files

## Tracker Outputs

Current options for the tracker outputs are built on Google's [Mediapipe]() and include pipelines for general [Pose](), [Hands](), and [Face]().
The [Holistic]() tracker combines all three outputs.
While the Holistic tracker offers improved tracking of the face and hands compared to the Pose model, the number of points it supplies can quickly become unweildy (several hundred for the face).
The Simple Holistic model filters out many of these points that may be extraneous to users primarily interested in gross skeletal movement. 


## Metarig Generation

For the Simple Holistic tracker you can generate a metarig configuration file. This will provide a set of parameters that can scale segments of a skeletal model based on the average distances between various landmarks throughout a dynamic calibration motion trial where the subject flexes and extends their joints with minimal camera occlusion.

With a more accurately scaled skeletal model, inverse kinematics can more successfully approximate the true movement.

## Practical Recording Guidelines


1. Minimize motion blur
    - motion blur can substantially compromise landmark recognition
    - using a higher frame rate can reduce motion blur
      - this will require more light to maintain good illumination


