# FAQ

### Can I process videos I pre-recorded with GoPros, phones, etc.?

Yes. Caliscope works with pre-recorded video from any camera, including GoPros and smartphones. Frame synchronization is required for extrinsic calibration and reconstruction. You can either ensure all cameras are hardware-synchronized (same trigger, same frame count) or provide a `timestamps.csv` file that records when each frame was captured. See [Project Setup](project_setup.md#timestampscsv-format) for the file format.

### How can I use something other than MediaPipe for tracking?

Caliscope supports custom pose estimation models in ONNX format. You can use models from RTMPose (MMPose), SLEAP, DeepLabCut, or any other framework that exports to ONNX. See [Custom ONNX Trackers](onnx_trackers.md) for setup instructions.

The built-in MediaPipe trackers (Hand, Pose, Simple Holistic, Holistic) are included for convenience. Any model that detects 2D landmarks in images can be integrated through the ONNX tracker system or by implementing the Tracker base class in Python.

### What calibration targets are supported?

Caliscope supports three target types:

- **ChArUco board**: works for both intrinsic and extrinsic calibration
- **Chessboard**: intrinsic calibration only
- **ArUco marker**: extrinsic calibration only

See [Calibration Targets](calibration_targets.md) for when to use each.

### Can the software export to Blender (or Unreal/Maya/etc.)?

Caliscope exports 3D trajectories in `.csv` and `.trc` formats. The `.trc` format is designed for biomechanical analysis in OpenSim. For Blender integration, an early-stage companion project called [Rigmarole](https://github.com/mprib/rigmarole) can create a scaled and animated rig from Caliscope's output.

### Does it do real-time processing?

Caliscope does not currently support real-time processing. Landmark tracking across multiple camera views is computationally intensive and is designed for offline batch processing.

### What is happening with my data? Are you storing videos I record?

No. Caliscope operates entirely on your local machine. No data is uploaded, transmitted, or stored externally. All video processing and calibration happens locally. This makes it suitable for clinical and research settings where data privacy is a concern.
