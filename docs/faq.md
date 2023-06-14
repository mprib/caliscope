# FAQ

### Is real-time operation possible?

While the data processing pipeline is designed with the ultimate goal of real-time tracking, the current version does not support it. The processing demands of landmark detection across concurrent frames currently throttles the frame rate to such an extent that I don't consider this a worthile investment of time at the moment. As a stack of hardware/tracking algorithm emerges that shows a viable path to a scaleable system, this will get bumped as a priority. If you have expertise in this area and are interested in contributing, please consider opening up a thread in the [discussions](https://github.com/mprib/pyxy3d/discussions) to start a conversation.

### Why is the software limited to Windows?

The sample markerless tracking tool used in this software, Google's [Holistic Mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md), is configured to run only on Windows and only on CPU. As my testing and development resources are currently limited to this system, I have not ventured to implement a cross-platform solution yet.

However, while the sample tracking tool may not run on other platforms, the core camera calibration probably will. The underlying tools, OpenCV/SciPy/PyQT, are cross-platform, but please anticipate some challenges as this has not been extensively tested.

### Can I use my smartphone as a camera?

Unfortunately, no. The project aims to provide an open-source, simple, scalable, and low-cost system. This involves heavy reliance on OpenCV for camera management. Currently, supporting input streams not directly managed by OpenCV presents a significant challenge and would detract from the development of core processes.

### Which webcam should I purchase?

Start with whatever webcam you currently have on hand. Conduct tests with two cameras to get a feel of how things run on your local system, and scale out from there. I have had success with the EMeet HD1080p cameras, which are reasonably priced (~$25 on Amazon). More expensive cameras with additional features such as autofocus have presented complications in my experience. If you have had a positive or negative experience with a specific webcam, kindly share it on our [discussions](https://github.com/mprib/pyxy3d/discussions) page.

### Can the software export to Blender (or Unreal/Maya/etc)?

Currently, the software only exports unfiltered 3D estimates in `csv` and `trc` formats. The `trc` format is designed for biomechanists. Those interested in creating an output pipeline to other formats may find the 'csv' files a good starting point.

### What is happening with my data? Are you storing videos I record?

Absolutely not. All operations are performed locally on your machine. 

### Can I process pre-recorded videos?

The software does not currently support the processing of pre-recorded videos, but this feature is planned for near-term future development. Processing videos offline will enable the use of more cameras and higher frame rates and resolutions, but it also requires some method of frame synchronization. It also presents the need to perform intrinsic calibration from pre-recorded videos. This is a developmpent priority, but not currently implemented. I aim to create an API that will support such post-processing in the future so that Pyxy3D could be used programmatically by Python processing pipelines.
