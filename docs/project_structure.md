## Current Flow

The general flow of processing is illustrated in the graph below. 


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
    StereoFrameBuilder
end

Synchronizer --SyncPacket-->  StereoFrameBuilder

LiveStream -.FramePacket.-> MonoCalibrator
MonoCalibrator -.Intrinsics.-> config.toml

VideoRecorder --> frame_time_history.csv
VideoRecorder --> port_X.mp4 
VideoRecorder -.During StereoFrameBuilder.-> point_data.csv
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

RecordingDirectory -.to be done.-> CaptureVolumeVisualizer

point_data.csv --> StereoCalibrator
config.toml --CameraSettings--> StereoCalibrator
StereoCalibrator -.StereoPairs.-> config.toml



subgraph calibration_data
    point_data.csv
    config.toml
end

point_data.csv --> get_stereotriangulated_table

ArrayTriangulator --> get_stereotriangulated_table

CornerTracker --PointPacket--> RecordedStream

ArrayConstructor --> CameraArray
config.toml --> ArrayConstructor

CameraArray --> ArrayTriangulator

subgraph triangulate
    ArrayTriangulator
    StereoPointsBuilder --- ArrayTriangulator
    StereoTriangulator --- ArrayTriangulator
end

CaptureVolume --> CaptureVolumeVisualizer

get_point_estimates  --> PointEstimates


subgraph capture_volume
    subgraph helper_functions
        get_stereotriangulated_table -.stereotriangulated_table DF.-> get_point_estimates   
    end

    CameraArray --> CaptureVolume
    PointEstimates --> CaptureVolume
    CaptureVolume --> QualityScanner
    QualityScanner -.filtered.-> PointEstimates
end


subgraph visualization
    CaptureVolumeVisualizer
    CameraMesh --> CaptureVolumeVisualizer
end

```
