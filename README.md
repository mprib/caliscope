## Current Flow

The general flow of processing is illustrated in the graph below. 

The `Synchronizer` is now producing `SyncPackets` from the set of  `LiveStream` objects provided to it. The previous code for recording video will no longer work, so must be updated. Additionally, when recording video the `VideoRecorder` should save out any point data that is calculated during the recording session so that it can be processed downstream.

The general plan for a revision to the current process is shown here.

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


Synchronizer --SyncPacket-->  OmniFrame
recording <--> RecordingDirectory

subgraph RecordingDirectory
port_P.mp4
frame_time_history.csv
point_data.csv
end

subgraph recording
VideoRecorder 
RecordedStream
end

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
