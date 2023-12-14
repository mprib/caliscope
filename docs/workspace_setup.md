# Workspace Setup


When a new project is created, the workspace will automatically populate the necessary folder structure if it does not already exist. There are 2 primary folders: `calibration` and `recordings`. Within `calibration` there must be subfolders for 'intrinsic' and 'extrinsic'. All motion capture trials must be stored seperately within subfolders of `recordings`.

An example initial project folder structure would therefore look like this:

# Initial Project Structure for calibration
```
ProjectDirectory/
├── calibration/
│   ├── intrinsic/
│   └── extrinsic/
└── recordings/

```

# Intrinsic Calibration Data

The first files to put in place are for the intrinsic calibration. These must follow the naming convention `port_1.mp4`, `port_2.mp4`, etc. They do not need to be synchronized. See [Best Practices] for getting a good calibration.

As the calibration commences, calculated variables are stored in 'config.toml' at the project root. Following the extrinsic calibration, an additional file called `point_estimates.toml` will be created. 




# Example of Project with Processed Motion Capture Trial
```
ProjectDirectory/
├── config.toml
├── point_estimates.toml
├── calibration/
│   ├── intrinsic/
│   │   ├── port_1.mp4   # These files do not need to be synchronized
│   │   ├── port_2.mp4   # Unsynchronized files
│   │   └── port_3.mp4   # Unsynchronized files
│   └── extrinsic/
│       ├── frame_time_history.csv  # Time reference for frame synchronization (optional)
│       ├── port_1.mp4              # Must be synchronized or use frame_time_history.csv
│       ├── port_2.mp4              # Must be synchronized or use frame_time_history.csv
│       └── port_3.mp4              # Must be synchronized or use frame_time_history.csv
└── recordings/
    └── recording_1/
        ├── frame_time_history.csv   # Time reference for frame synchronization (optional)
        ├── port_1.mp4               # Must be synchronized or use frame_time_history.csv
        ├── port_2.mp4               # Must be synchronized or use frame_time_history.csv
        ├── port_3.mp4               # Must be synchronized or use frame_time_history.csv
        └── HOLISTIC/                # Output subfolder with visualized landmarks
            ├── frame_time_history.csv            # Matches file in parent folder
            ├── port_0_HOLISTIC.mp4               # Copy of file in parent folder with visualized landmarks
            ├── port_1_HOLISTIC.mp4               # Copy of file in parent folder with visualized landmarks
            ├── port_2_HOLISTIC.mp4               # Copy of file in parent folder with visualized landmarks
            ├── xy_HOLISTIC.csv                   # Raw output of all points
            ├── xyz_HOLISTIC.csv                  # Triangulated output by point ID
            ├── xyz_HOLISTIC_labelled.csv         # Triangulated output in tidy format with labelled x, y, z columns
            └── xyz_HOLISTIC.trc                  # Landmark data for OpenSim

```
