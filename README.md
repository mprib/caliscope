## Current Flow

The general flow of processing is illustrated in the graph below. 

Incremental improvements in the flow of the information processing are reflected below. The primary change for this branch is that they synchronizer will push a SyncPacket to the StereoTracker. The StereoTracker will then use the SyncPacket create a PairedPointsPacket. This will provide the primary input for the OmniFrame. One aspect of this that I have not sorted out is where the history of points used by the OmniFrame for feedback to the user will be stored.

```mermaid
graph TD


LiveStream --FramePacket--> Synchronizer
RecordedStream --FramePacket--> Synchronizer
Synchronizer --> VideoRecorder

subgraph cameras
Camera --> LiveStream
end
subgraph tracking
Charuco --> CornerTracker
CornerTracker --PointPacket--> LiveStream
end

subgraph recording
VideoRecorder --> RecordingDirectory
RecordingDirectory --> RecordedStream
end

Synchronizer --SyncPacket-->  OmniFrame


subgraph calibration_data
config.toml
StereoCalRecordings
end

CornerTracker --PointPacket--> RecordedStream

calibration -.via Session.-> StereoCalRecordings
calibration_data --> ArrayConstructor

subgraph array
ArrayConstructor
end



CornerTracker -.temporary for testing.- PairedPointStream

subgraph triangulate
PairedPointStream --> StereoTriangulator
end

subgraph visualization
CameraMesh --> Visualizer
StereoTriangulator --> Visualizer
end

```
