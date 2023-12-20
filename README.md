
NOTE (12/19/2023): The core [docs](https://mprib.github.io/pyxy3d/) have recently been revised to align with the new pre-recorded footage workflow. I will be expanding documentation and demonstration videos in the coming week. If you have any interest in this project, please consider poking around and providing any feedback you may have. I (mprib) am at a point where I really need feedback from early stage users to know how to best prioritize the time that I can dedicate to this project.

<div align="center"><img src = "pyxy3d/gui/icons/pyxy_logo.svg" width = "150"></div>

<div align="center">
  
</div>




## About

Pyxy3d (*pixie-3d*) is a **py**thon package that integrates:

- multicamera calibration
- 2D (**x,y**) landmark tracking
- **3D** landmark triangulation. 

It is GUI-based, permissively licensed under the LGPLv3, and intended to serve as the processing hub of a low-cost DIY motion capture studio.

Currently is uses Google's Mediapipe for markerless tracking, though is built on a Tracker API that can be used to integrate alternate landmark tracking tools.

Please see our [docs](https://mprib.github.io/pyxy3d/) for details about installation, project setup, and general workflow.

### Demo Output
---
https://github.com/mprib/pyxy3d/assets/31831778/803a4ce8-4012-4da1-87b9-66c5e6b31c59

*The above was created using Pyxy3D, a 7 year old t440p laptop, and 4 webcams (~$25 each). This includes camera calibration, recording of synchronized frames (720p @ 24 fps), landmark detection, and point triangulation. Animated rig creation was done using an early stage Blender add-on project called [Rigmarole](https://github.com/mprib/rigmarole)*

---

## Quick Start


## Key Features



## Limitations

### Requires Frame Sync
The workflow currently requires you to provide your own synchronized frames or to provide [a file](project_setup.md#frame_time_historycsv) that specifies the time at which each frame was read so that pyxy3d can perform the synchronization itself. There are plans to manage this synchronization automatically through audio files, though that has not yet been implemented.

### Currently only using Mediapipe

Google's Mediapipe provides a relatively easy and efficient method for human subject tracking, though for many uses it is limiting. Pyxy3D has a general Tracker base class that is implemented in a few versions (Pose/Hands/Holistic). This has provided a proof of concept implementation of markerless tracking, though for more robust use the roadmap calls for integration with more powerful tools such as [MMPose](https://github.com/open-mmlab/mmpose) and [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut).

## Reporting Issues and Requesting Features

To report a bug or request a feature, please [open an issue](https://github.com/mprib/pyxy3d/issues). Please keep in mind that this is an open-source project supported by volunteer effort, so your patience is appreciated.

## General Questions and Conversation

Post any questions in the [Discussions](https://github.com/mprib/pyxy3d/discussions) section of the repo. 


## Acknowledgments

This project was inspired by [FreeMoCap](https://github.com/freemocap/freemocap) (FMC), which is spearheaded by [Jon Matthis, PhD](https://jonmatthis.com/) of the HuMoN Research Lab. The FMC calibration and triangulation system is built upon [Anipose](https://github.com/lambdaloop/anipose), created by Lili Karushchek, PhD. Pyxy3D was originally envisioned as an alternative calibration tool to Anipose that would allow more granular estimation of intrinsics as well as visual feedback during the calibration process. Several lines of of the original Anipose triangulation code are used in this code base, though otherwise it was written from the ground up. I'm grateful to Dr. Matthis for his time developing FreeMoCap, discussing it with me, pointing out important code considerations, and providing a great deal of information regarding open-source project management.

## License

Pyxy3D is licensed under [LGPL-3.0](https://www.gnu.org/licenses/lgpl-3.0.html). The triangulation function was adapted from the [Anipose](https://github.com/lambdaloop/anipose) code base which is licensed under [BSD-2 Clause](https://opensource.org/license/bsd-2-clause/).
