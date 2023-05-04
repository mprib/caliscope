# Development Notes

Currently this is going to just serve as a catch-all for things that I think are important to note and that I'm afraid I will forget. 


## Storage of (x,y) point data
Storage of xy.csv datapoints can take place in either of two ways: 
1. During live recording the VideoRecorder will store xy points when a tracker is running live (i.e. there are points in the frame packets). This is currently *only* used as a way to avoid re-processing videos after the fact so that the calibration will occur more quickly. 
2. During playback of a recorded stream when provided with a tracker. 

In both instances, points will be saved to `xy.csv` in the same directory that the recorded videos are saved. 

### *a note explaining this choice*

I had considered pushing the save feature into the LiveStream so that there would be more similarity between the two types. A complication exists which is how to signal that the recording should start and stop. 

I'm realizing now that it may make sense for me to push the recording feature into the stream itself, which would remove the need for a dedicated Recorder, and provides a simpler path to more modular processing pipelines that could exist independently on different machines. This may be a step for the future.  