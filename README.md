<div align="center">

# Caliscope

*Multicamera Calibration + Pose Estimation --> Open Source Motion Capture*

[![PyPI - Downloads](https://img.shields.io/pypi/dm/caliscope?color=blue)](https://pypi.org/project/caliscope/)
[![PyPI - License](https://img.shields.io/pypi/l/caliscope?color=blue)](https://opensource.org/license/bsd-2-clause/)
[![PyPI - Version](https://img.shields.io/pypi/v/caliscope?color=blue)](https://pypi.org/project/caliscope/)
[![GitHub last commit](https://img.shields.io/github/last-commit/mprib/caliscope.svg)](https://github.com/mprib/caliscope/commits)
[![GitHub stars](https://img.shields.io/github/stars/mprib/caliscope.svg?style=social&label=Star)](https://github.com/mprib/caliscope/stargazers)
![pytest](https://github.com/mprib/caliscope/actions/workflows/pytest.yml/badge.svg)
</div>

Caliscope is a GUI-based multicamera calibration package.
It simplifies the process of determining camera properties to enable 3D motion capture.

## Quick Start

Basic installation instructions can be found in our [docs](https://mprib.github.io/caliscope/installation/).
Please note that installation can take a while due to large dependencies like OpenCV and PySide6.

For a complete overview of the entire workflow, please see the [sample project](https://mprib.github.io/caliscope/sample_project/).
A [video walk through](https://www.youtube.com/watch?v=voE3IKYtuIQ) demonstrates the process with an example dataset.

## Demo of Core Features

### Calibration Board Creation and Camera Intrinsic Calibration

https://github.com/user-attachments/assets/c2dd4119-772a-4076-90f7-4e6201f604ed


### Estimate Multicamera Relative Pose and Set World Origin


https://github.com/user-attachments/assets/6e21c5bb-b8d1-4999-88f8-735bb5722570




### Integrate with Tracking Tools To Triangulate Landmarks



### Demo Animation

https://github.com/mprib/caliscope/assets/31831778/803a4ce8-4012-4da1-87b9-66c5e6b31c59

*`Caliscope` was used to calibrate the cameras (both intrinsic and extrinsic parameters) and triangulate the 3D landmark positions shown above.*
*The 2D landmark estimation was run across all videos using Google's Holistic Mediapipe.*

---


---

## How It Works

To triangulate 3D landmarks from synchronized video, you must know the intrinsic and extrinsic properties of your camera system.
Intrinsic properties include each camera's focal length, optical center, and lens distortion.
Extrinsic properties describe the relative rotation and translation of all cameras in the system.
Using more cameras makes 3D tracking more robust to occlusion and other inevitable errors.

While OpenCV provides functions for single-camera intrinsics, estimating extrinsics for more than two cameras is not straightforward.
This multi-camera process requires a technique called [bundle adjustment](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html), which demands extensive tracking of camera parameters and 2D point estimates.

Caliscope automates this calibration process from only raw video and a definition of your calibration board.
It provides visual feedback at each stage, helping you verify the parameter estimates.

## Key Features

#### Calibration

- Easy creation of `png` files for ChArUco calibration boards.
- Automated calculation of camera intrinsic properties from input video.
- Visualization of the distortion model to ensure reasonableness.
- Automated bundle adjustment to estimate the 6-DoF relative position of all cameras.
- A 3D visualizer to inspect camera position estimates.
- Tools to set the world origin within the visualizer to simplify data processing.

#### 3D Tracking



- A general Tracker interface for integrating alternate 2D tracking methods.
- Three sample implementations using Google Mediapipe (Hands/Pose/Holistic).
- Automated application of 2D landmark tracking to synchronized videos.
- Triangulation of 3D landmark positions based on the full camera system calibration.
- Trajectory smoothing through gap-filling and Butterworth filtering.

#### Data Export

- Output to the `.trc` file format for use in biomechanical modeling.
- Output to a tidy `.csv` format for integration with other analysis workflows.

## Roadmap & Integrations

The current tracker implementations provide a proof-of-concept pipeline using Google's Mediapipe.
While Mediapipe is an easy and efficient method for human tracking, it has limitations in accuracy and precision.
The planned roadmap includes integration with more powerful tools like [MMPose](https://github.com/open-mmlab/mmpose), [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), and [SLEAP](https://github.com/talmolab/sleap).

## Community & Support

To report a bug or request a feature, please [open an issue](https://github.com/mprib/caliscope/issues).
Please keep in mind this is an open-source project supported by volunteer effort, so your patience is appreciated.

For general questions and conversation, please post in the [Discussions](https://github.com/mprib/caliscope/discussions) section of the repo.

## Acknowledgments

This project was inspired by [FreeMoCap](https://github.com/freemocap/freemocap) (FMC), which is spearheaded by [Jon Matthis, PhD](https://jonmatthis.com/) of the HuMoN Research Lab.
The FMC calibration and triangulation system is built upon [Anipose](https://github.com/lambdaloop/anipose), created by Lili Karushchek, PhD.
Caliscope was originally envisioned as an alternative calibration tool to Anipose that would allow more granular estimation and visual feedback.

Several lines of the original Anipose triangulation code are used in this code base, though it was otherwise written from the ground up.
I'm grateful to Dr. Matthis for his time developing FreeMoCap, discussing it with me, and providing a great deal of information regarding open-source project management.

## License

Caliscope is licensed under the permissive [BSD 2-Clause license](https://opensource.org/license/bsd-2-clause/).
The triangulation function was adapted from the [Anipose](https://github.com/lambdaloop/anipose) code base which is also licensed under the BSD-2 Clause.
A primary dependency of this project is PySide6 which provides the GUI front end.
PySide6 is licensed under the [LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.html).
Caliscope does not modify the underlying source code of PySide6 which is available via [PyPI](https://pypi.org/project/PySide6/).
