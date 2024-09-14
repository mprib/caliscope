# Video Capture

Synchronized video footage for extrinsic calibration as well as motion capture can be obtained from specialized camera systems (such as [FLIR](https://www.flir.com/support-center/iis/machine-vision/application-note/configuring-synchronized-capture-with-multiple-cameras/)).

As a low-cost alternative, a companion project called [MultiWebCam](https://github.com/mprib/multiwebcam) was developed that can allow concurrent recording of frames from multiple webcams, along with time stamps of when the frames were read from the camera. This allows for the frames to be roughly time-aligned in post processing.