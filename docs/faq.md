# FAQ

### Can I process videos I pre-recorded with GoPros, some phones, etc?

Yes. Note that the frame data must be synchronized for calibration and triangulation. An automated pipeline for synchronizing frames using audio data is part of the active roadmap. Until that emerges, you will need to provide your own timestamp data in the specified format.

### How can we use something other than Mediapipe for tracking?

A priority in the project roadmap is to build out integration with alternate tracking tools such as MMPose and DeepLabCut. Mediapipe variations (hand,pose,holistic) are currently implemented both becuase they can efficiently run on CPU and because they provide examples for how to implement the underlying Tracker base class that is used through the pyxy3d pipeline. If you can create a python object that will take a frame as an input and provide points as an output, then it can be integrated into the entire workflow by following the Tracker API. 

### Can the software export to Blender (or Unreal/Maya/etc)?

Currently, the software only exports 3D estimates in `csv` and `trc` formats. The `trc` format is designed for biomechanists. There is a *very* early stage companion project called Rigmarole that is intended to become a Blender plug-in. Rigmarole will generate an appropriately scaled and animated rig from the pyxy3d standard output. This code was used to automate the creation of the ballet dancer on the main page of the repo.

### Does it do real-time processing?

No. Landmark tracking is computationally intensive, and doing so on multiple viewpoints while concurrently reading synchronized frames from multiple cameras will quickly max out a standard desktop. It is certainly possible to perform this kind of real-time tracking and triangulation (see: demo), but given current hardware it will have limited spatial and temporal resolution, as well as limited redundency of views. It is (currently) not a priority or part of the roadmap.

### What is happening with my data? Are you storing videos I record?

Absolutely not. All operations are performed locally on your machine. An imagined future use-case for this package is as a tool that could be used in clinical settings or human subjects research. Data privacy is absolutely critical under those circumstances. The commitment that **you will always control your data** is at the heart of this project. 
