# Overview

A moderate scale refactor is underway impacting the monocalibrator. Single camera calibration as well as corner tracking was deeply nested within the video stream object and I have been working to create more loosly coupled classes with better internal coherence. 

Side effects of this refactor include the need to push more responsibility onto the session object in order to maintain interaction between the increasing number of classes. This manifests in the GUI braking.

## Current Branch Targets

Refactor CameraConfigDialog to operate when provided only with a monocalibrator. 

Monocalabrator may need to be changed to create a queue for exposing the grid_frame. This could be passed to the frame_emitter