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

subgraph GUI
    MonoCalibrator
    OmniFrame
end

Synchronizer --SyncPacket-->  OmniFrame

LiveStream -.FramePacket.-> MonoCalibrator
MonoCalibrator -.Intrinsics.-> config.toml

VideoRecorder --> frame_time_history.csv
VideoRecorder --> port_X.mp4 
VideoRecorder -.During OmniFrame.-> point_data.csv
port_X.mp4 --> RecordedStream
frame_time_history.csv --> RecordedStream


subgraph recording

    RecordedStream
    VideoRecorder 

    subgraph RecordingDirectory
        port_X.mp4
        frame_time_history.csv
    end

end



point_data.csv --> OmniCalibrator
config.toml --CameraSettings--> OmniCalibrator
OmniCalibrator -.StereoPairs.-> config.toml



subgraph calibration_data
    point_data.csv
    config.toml
end


CornerTracker --PointPacket--> RecordedStream

subgraph array
    config.toml --> ArrayConstructor
    ArrayConstructor --> CameraArray
end

CameraArray --> Visualizer
CameraArray --> StereoTriangulator
Synchronizer -.FramePacket.-> PairedPointStream

subgraph triangulate
    PairedPointStream -.PairedPointPacket.-> ArrayTriangulator
    StereoTriangulator
end

StereoTriangulator --> ArrayTriangulator
ArrayTriangulator -.To Be Done.-> triangulated_points.csv
triangulated_points.csv --> Visualizer

subgraph visualization
    CameraMesh --> Visualizer
end

```
