# Post Processing Pipeline

Following recording, the `mp4` files, augmented by `frame_time_history.csv` and `config.toml` (needed for camera rotation count), are used to generate (x,y) coordinates via the `Tracker`. This is generally the most time consuming portion of the post-processing. A subdirectory will be created that contains the (x,y) file as well as the video file swith frame points displayed.

This (x,y) file can then be used along with the `config.toml` (for the camera intrinsics and extrinsics) to generate `xyz_TRACKER.csv` within this `TRACKER` subdirectory.

The (x,y,z) points here are only identified via point ID. The `Tracker` can be used to generate a labelled file in wide format. This labelled version of the file can then be used to generate a `trc` file which can be used by OpenSim. 

It's not intended for users to manage this granular file production themselves, but this documentation is intended for development record keeping and to provide context on the project file structure and multitude of files produced.

The following shows the layout of a 3 camera project file with 1 recording where it has been processed by a HOLISTIC tracker with output to a trc file for use in OpenSim.

```
Project/
├── calibration
├── config.toml
└── recording_1
    ├── frame_time_history.csv
    ├── port_0.mp4
    ├── port_1.mp4
    ├── port_2.mp4
    └── HOLISTIC
        ├── frame_time_history.csv
        ├── port_0_HOLISTIC.mp4
        ├── port_1_HOLISTIC.mp4
        ├── port_2_HOLISTIC.mp4
        ├── xy_HOLISTIC.csv
        ├── xyz_HOLISTIC.csv
        ├── xyz_HOLISTIC_labelled.csv
        └── xyz_HOLISTIC.trc
```