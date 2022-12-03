# Overview

Individual and stereo camera calibration is now operating within a PyQt GUI. The next stage of development is to integrate the calibration data into a single unified system that can be allow integration of all stereolocated point locations.

Before tackling the larger task, I will take a smaller step by doing 3d tracking of a charuco board fron one stereocamera perspective.

## Next Step

Up to this point in the process, iterations have been slowed by the fact that I am collecting live video data each time I trial run some element of the calibration. Now I would like to switch over to using pre-recorded video, particularly now that the calibration (and need for real time feedback) has been largely locked down.

So the question is how to write data to files in a way that can later be played back and processed.
