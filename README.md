# Overview

A moderate scale refactor is underway impacting the monocalibrator. Single camera calibration as well as corner tracking was deeply nested within the video stream object and I have been working to create more loosly coupled classes with better internal coherence. 

Side effects of this refactor include the need to push more responsibility onto the session object in order to maintain interaction between the increasing number of classes. This manifests in the GUI braking.

## Current Branch Targets

Build out the session class to create a syncronizer and dispatcher. Monocalibrators can be loosely spun up by the GUI to hook into the dispatcher and camera. I believe this stage of the change will be challenging, but ultimately lead to a much better framework for moving forward. 