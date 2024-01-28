# Workspace Setup


## Initial Project Structure
When a new project is created, the workspace will automatically populate the necessary folder structure if it does not already exist. There are 2 primary folders: `calibration` and `recordings`. Within `calibration` there must be subfolders for `intrinsic` and `extrinsic`. All motion capture trials must be stored separately within subfolders of `recordings` by the user.

An configuration file called `config.toml` will be automatically created when a new project is created. Initially this will only be storing the default charuco board definition. 

An example initial project folder structure would therefore look like this:
```
ProjectDirectory/
├── config.toml    # Only contains default charuco board definition
├── calibration/
│   ├── intrinsic/
│   └── extrinsic/
└── recordings/    # Empty by default prior to user populating data
```

## Stage 1: Intrinsic Calibration

Place video files for the intrinsic camera calibration in the `intrinsic` folder. 

These must follow the naming convention `port_1.mp4`, `port_2.mp4`, etc. They do not need to be synchronized. 

A project with 3 cameras would therefore look something like this going into the intrinsic camera calibration. 

```
ProjectDirectory/
├── config.toml          # following intrinsic calibration, this file will also have the camera matrix and distortion for each source camera
├── calibration/
│   ├── intrinsic/
│   │   ├── port_1.mp4   # These files do not need to be synchronized
│   │   ├── port_2.mp4   # Unsynchronized files
│   │   └── port_3.mp4   # Unsynchronized files
│   └── extrinsic/
└── recordings/
```

As the intrinsic properties of the camera are calculated, parameters are stored in `config.toml` at the project root.


## Stage 2: Extrinsic Calibration

Place sychronized video files in the `extrinsic` folder. Synchronization can be accomplished in one of two ways:

1. Record all video footage with a common external trigger such that each frame is at the same point in time as the corresponding frames from the other files. In other words: all mp4 files should start and stop at the same moment in time and have the same number of frames.

2. Provide a file called `frame_time_history.csv` within the folder of recorded video. This must provide the time at which each frame was read so that caliscope can synchronize the footage during processing. 
   
   
### `frame_time_history.csv`

This file will have a structure like this:

```
port,frame_time
3,927387.33536115
4,927387.50128975
1,927387.3530109001
3,927387.50643105
1,927387.51819965
2,927387.5063038999
3,927387.6684489499
4,927387.66848565
3,927387.8359558999
4,927387.8360615501
...
```

It does not need to be in any special order. The numbers shown above are from `time.perf_counter()` in the standard python library, but any numerical value that shows the relative time of the frame reads will work. There do not need to be the same number of frames within each `mp4` file. They do not need to start on the same frame. The synchronization will take place automatically, including inserting a blank frame when necessary to keep the video streams aligned in time.


### Final file structure following extrinsic calibration

Following the extrinsic calibration, an additional file called `point_estimates.toml` will be created. This contains data used to estimate the relative camera translations and rotations. A project with fully calibrated extrinsics would thus look something like this:


```
ProjectDirectory/
├── config.toml          # Now contains rotation and translation parameters for each camera in addition to the distortion and camera matrix
├── point_estimates.toml # Contains charuco data used to estimate the relative camera positions
├── calibration/
│   ├── intrinsic/       # directory unchanged from above
│   │   ├── port_1.mp4   
│   │   ├── port_2.mp4   
│   │   └── port_3.mp4   
│   └── extrinsic/
│       ├── frame_time_history.csv  # Time reference for frame synchronization (optional)
│       ├── port_1.mp4              # Must be synchronized or use frame_time_history.csv
│       ├── port_2.mp4              # Must be synchronized or use frame_time_history.csv
│       └── port_3.mp4              # Must be synchronized or use frame_time_history.csv
└── recordings/
```


## Stage 3: Processing Motion Capture Trial

For each motion capture trial, create a subfolder within `recordings` and populate it with synchronized footage as was done with the extrinsic calibration. After post-processing of the video footage has occurred, output will be created as shown in the following example:

```
ProjectDirectory/
├── config.toml                             # File unchanged from above
├── point_estimates.toml                    # File unchanced from above
├── calibration/                            # Entire calibration directory unchanged from above
│   ├── intrinsic/
│   │   ├── port_1.mp4   
│   │   ├── port_2.mp4   
│   │   └── port_3.mp4   
│   └── extrinsic/
│       ├── frame_time_history.csv  
│       ├── port_1.mp4              
│       ├── port_2.mp4              
│       └── port_3.mp4              
└── recordings/
    └── recording_1/                              # can be named anything; contents follow formatting of extrinsic calibration folder
        ├── frame_time_history.csv                # optional file; not needed if all video synchronized frame-for-frame
        ├── port_1.mp4                      
        ├── port_2.mp4                      
        ├── port_3.mp4                      
        └── HOLISTIC/                             # Output subfolder created when running Holistic Mediapipe Tracker 
            ├── frame_time_history.csv            # Matches file in parent folder
            ├── port_0_HOLISTIC.mp4               # Copy of file in parent folder with visualized landmarks
            ├── port_1_HOLISTIC.mp4               # Copy of file in parent folder with visualized landmarks
            ├── port_2_HOLISTIC.mp4               # Copy of file in parent folder with visualized landmarks
            ├── xy_HOLISTIC.csv                   # All 2D tracked points by source and point id
            ├── xyz_HOLISTIC.csv                  # Triangulated output by point ID
            ├── xyz_HOLISTIC_labelled.csv         # Triangulated output in tidy format with labelled x, y, z columns
            └── xyz_HOLISTIC.trc                  # Triangulated Landmark data for OpenSim
```
