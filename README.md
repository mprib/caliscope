# Current Functionality

To launch the primary functionality of the repo, run `src\gui\main.py`. This will open an ugly dialog to begin a calibration session. Session data is stored in the `sessions` folder in the root directory. The session being launched is coded at the bottom of `main.py`. The general workflow:

1. Print out a charuco board from the Charuco Builder tab
2. Click "Find Additional Cameras" to connect to them
3. Go to the camera tabs and configure resolution/exposure before collecting calibration corners and calibrating.
4. Save calibration parameters on camera tab
5. Once all cameras intrinsic factors are estimated, click the stereocalibration button and calibrate each pair of cameras.
6. Save calibration to `config.toml`.

# Recording

The file `src\recording\video_recorder.py` will launch a VideoRecorder and store the mp4 and frametime data in the session folder. 

# Triangulating

Recorded video can be played back through the synchronizer with synched frame bundles passed to the PairedPointStream which uses the charuco tracker to identify the same point in space between paired frames. Point ID and (X,Y) position for each frame are passed to an `out_q`. The triangulator pulls from this queue, calculates values for ID: (X,Y,Z) and places that calculation on its own `out_q`

# Visualization

The Visualizer constructs camera meshes based on `config.toml` to allow a gut check of the stereocalibration parameters. The camera mesh shape is determined by the actual camera properties and provides another way of assessing reasonableness at a glance. (X,Y,Z) points are read from `triangulator.out_q` and updated to the scene. 

Currently, the origin is the base camera from the stereocalibration pair. Next planned steps include a way to set the origin.


## Things to Uninstall

Going back to Temuge B notebook I instaled plotly and something called nbformat just to visualize the output. 

Don't carry these packages forward as they are not actually necessary for core work.

## Current Object Relationships

I need to see where everything is at the moment. As I go back to address issues with the scale of the charuco board impacting the scale of the calibration output data and how that is reflected in the visualization of the data, I'm realizing I need to take a step back and look at things more broadly.
```mermaid
graph

subgraph calibration
Charuco --> CornerTracker
CornerTracker --> Monocalibrator
CornerTracker --> Stereocalibrator
end

subgraph cameras
Camera -.only for writing settings.- Monocalibrator
Camera --> LiveStream
LiveStream --> Synchronizer
Synchronizer --> Monocalibrator
Synchronizer --> Stereocalibrator
end

Synchronizer --> PairedPointStream 

subgraph recording
Synchronizer --> VideoRecorder
VideoRecorder --> port_#.mp4
VideoRecorder --> frame_time_history.csv

port_#.mp4 --> RecordedStream
frame_time_history.csv --> RecordedStream
RecordedStream --> Synchronizer
end

CornerTracker -.temporary for testing .- PairedPointStream

subgraph triangulate
PairedPointStream -->StereoTriangulator
end

subgraph visualization
CameraMesh --> Visualizer
StereoTriangulator --> Visualizer
end

```
