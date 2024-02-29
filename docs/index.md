
# Welcome

Caliscope is a GUI-based and permissively licensed multicamera calibration package that integrates with 2D landmark tracking tools to produce low-cost, open-source motion capture.

The package includes sample markerless trackers using variations of Google's Mediapipe (hands/pose/holistic) which illustrate how to implement the underlying Tracker base class. The intention is to allow alternate tracking algorithms to be cleanly plugged into the pipeline.

The workflow currently requires you to provide your own synchronized frames or to provide [a file](project_setup.md#frame_time_historycsv) that specifies the time at which each frame was read so that caliscope can time-align the frames itself. A companion project is currently in development ([multiwebcam](https://github.com/mprib/multiwebcam)) that can perform concurrent USB webcam video for use cases where differences in frame capture of a few hundreds of a second are tolerable.

The [installation](installation.md) guide will walk you through the process of installing the package on your system. [Project Setup](project_setup.md) will show you the format for saving files so that they can be used. The workflow guides to the left will provide details about how to create a ChArUco board, calibrate the cameras (both intrinsic and extrinsic) and perform 3D landmark tracking from motion capture trials.

This project is at a very early stage so please bear with us while going through the inevitable growing pains that are ahead. You feedback is appreciated. If you have specific recommendations, please consider creating an [issue](https://github.com/mprib/caliscope/issues). If you have more general questions or thoughts about the project, please open up a thread in the [discussions](https://github.com/mprib/caliscope/discussions).

