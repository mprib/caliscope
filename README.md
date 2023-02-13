## Current Flow

The general flow of processing is illustrated in the graph below. 

The `Synchronizer` is now producing `SyncPackets` from the set of  `LiveStream` objects provided to it. The previous code for recording video will no longer work, so must be updated. Additionally, when recording video the `VideoRecorder` should save out any point data that is calculated during the recording session so that it can be processed downstream.

The general plan for a revision to the current process is shown here.

Note that the sections of code that do not have a link in some way to the synchronizer are not currently functional.

```mermaid
graph TD


LiveStream --FramePacket--> Synchronizer
RecordedStream --FramePacket--> Synchronizer
Synchronizer --SyncPacket--> VideoRecorder

subgraph cameras
Camera --> LiveStream
end
subgraph tracking
Charuco --> CornerTracker
CornerTracker --PointPacket--> LiveStream
end


Synchronizer --SyncPacket-->  OmniFrame

subgraph recording
RecordedStream
VideoRecorder 
end


VideoRecorder --> frame_time_history.csv
VideoRecorder --> port_X.mp4 
VideoRecorder --> point_data.csv

subgraph RecordingDirectory
port_X.mp4 --> RecordedStream
frame_time_history.csv --> RecordedStream
end


point_data.csv --> BulkMonocalibrator
config.toml --CameraSettings--> BulkMonocalibrator
BulkMonocalibrator -.Intrinsics.-> config.toml


subgraph calibration_data
point_data.csv
config.toml
StereoCalRecordings
end


CornerTracker --PointPacket--> RecordedStream

calibration -.via Session.-> StereoCalRecordings

subgraph array
ArrayConstructor
end



subgraph triangulate
PairedPointStream --> StereoTriangulator
end

subgraph visualization
CameraMesh --> Visualizer
StereoTriangulator --> Visualizer
end

```
