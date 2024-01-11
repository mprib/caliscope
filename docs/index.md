<div align="center"><img src = "images/pyxy3d_logo.svg" width = "150"></div>


# Welcome

Pyxy3D (*pixie-3D*) is intended to serve as the core software component of a low-cost DIY motion capture studio. It is **py**thon package that integrates:

- multicamera calibration
- 2D (**x,y**) landmark tracking
- **3D** landmark triangulation. 

It is GUI-based and permissively licensed under the BSD 2-Clause license.

The packages includes sample markerless trackers using variations of Google's Mediapipe (hands/pose/holistic) which illustrate how to implement the underlying Tracker base class. The intention is to allow alternate tracking algorithms to be cleanly plugged into the pipeline.

The workflow currently requires you to provide your own synchronized frames or to provide [a file](project_setup.md#frame_time_historycsv) that specifies the time at which each frame was read so that pyxy3d can perform the synchronization itself, though there are plans to manage this synchronization automatically through audio files.

The [installation](installation.md) guide will walk you through the process of installing the package on your system. [Project Setup](project_setup.md) will show you the format for saving files so that they can be used. The workflow guides to the left will provide details about how to create a ChArUco board, calibrate the cameras (both intrinsic and extrinsic) and perform 3D landmark tracking from motion capture trials.

This project is at a very early stage so please bear with us while going through the inevitable growing pains that are ahead. You feedback is appreciated. If you have specific recommendations, please consider creating an [issue](https://github.com/mprib/pyxy3d/issues). If you have more general questions or thoughts about the project, please open up a thread in the [discussions](https://github.com/mprib/pyxy3d/discussions).

