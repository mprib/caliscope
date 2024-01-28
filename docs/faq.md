# FAQ

### Can I process videos I pre-recorded with GoPros, some phones, etc?

Absolutely! Caliscope supports processing of pre-recorded videos, including those from GoPros and smartphones. Keep in mind that frame synchronization is crucial for accurate calibration and triangulation. We're actively working on an automated solution to synchronize frames using audio data. For now, you'll need to provide your own timestamp data in the specified format or ensure that all video sources are synchoronized while the recording is being made.

### How can we use something other than Mediapipe for tracking?

We are prioritizing the integration of alternative tracking tools like MMPose and DeepLabCut into our project roadmap. While Mediapipe variations (hand, pose, holistic) are currently implemented for their simple hardware implementation and as examples for the Tracker base class, any Python object that accepts a frame and outputs points can be integrated. By adhering to the Tracker API, you can easily plug into the entire Caliscope workflow.

### Can the software export to Blender (or Unreal/Maya/etc)?

As of now, Caliscope exports 3D estimates in `csv` and `trc` formats, with `trc` being tailored for biomechanics. An early-stage companion project, [Rigmarole](https://github.com/mprib/rigmarole), aims to develop a Blender plugin that creates a scaled and animated rig from Caliscope's outputs. This plugin is in early development but was instrumental in creating the animated ballet dancer showcased on our main repo page.

### Does it do real-time processing?

Currently, Caliscope does not support real-time processing. Landmark tracking across multiple camera views is resource-intensive and challenging to execute in real-time on standard hardware. While real-time tracking and triangulation are feasible (as demonstrated in our demos), they come with limitations in resolution and redundancy of views. Enhancing real-time capabilities is not a current roadmap priority.

### What is happening with my data? Are you storing videos I record?

Absolutely not. Caliscope operates entirely locally on your machine. Envisioning future use in clinical or research settings, we prioritize data privacy. Your control over your data is fundamental to our commitment, ensuring all operations are confined to your local environment.
