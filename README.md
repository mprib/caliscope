# Overview

Individual and stereo camera calibration is now operating within a PyQt GUI. The next stage of development is to integrate the calibration data into a single unified system that can be allow integration of all stereolocated point locations.

Before tackling the larger task, I will take a smaller step by doing 3d tracking of a charuco board fron one stereocamera perspective.


# Recording Synched Video

To avoid repeated data acquisition during development, I need a way to just run off of recorded video. Because a long-term goal is to have real-time data processing, it would be best if all post-processing could occur through the same synchronizer interface, so the system will be set up to allow playback of recorded sessions through the synchronizer via PlaybackStreams which mimic the interface of the regular VideoStream.

## Next Step

The recording of the video has been completed. Now the PlaybackStream should be created to interface with the synchronizer. That is the objective of the current development branch (not currently reflected in Main).

