## Current Flow

The general flow of processing is illustrated in the graph below. 

This graph needs to be updated to reflect the centrality of the ArrayPointsErrorData dataclass in the post-processing analysis. This is turning into a useful construct that can summarize the complete state of the model and tracking data, while also containing the data necessary to relate each 3d projected point to the precice frames it came from.

It is quite tightly coupled to the camera array object. I would like to refactor to decouple these things into seperate modules, but the best way to do that isn't immediately clear to me. It may ultimately be the case that the bundle adjustment function is going to need to come out of the camera array, and go into something like an Optimizer that could manage both the camera array optimization during calibration, and the 3d point optimization during general triangulation.

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

point_data.csv --> SyncPacketBuilder
SyncPacketBuilder -.SyncPacket.-> StereoPointBuilder 



CornerTracker --PointPacket--> RecordedStream

ArrayConstructor --> CameraArray
config.toml --> ArrayConstructor

CameraArray --> ArrayTriangulator

subgraph triangulate
    StereoPointBuilder -.SynchedStereoPointsPacket-->ArrayTriangulator
    ArrayTriangulator
    StereoTriangulator
end

StereoTriangulator --> ArrayTriangulator
ArrayTriangulator --> triangulated_points.csv
CaptureVolume --> CaptureVolumeVisualizer

triangulated_points.csv --> PointEstimateData

subgraph capture_volume
CameraArray --> CaptureVolume
PointEstimateData --> CaptureVolume

CaptureVolume
end



subgraph visualization
    CaptureVolumeVisualizer
    CameraMesh --> CaptureVolumeVisualizer
end

```
