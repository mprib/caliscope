
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


## About
`Caliscope` is a GUI-based multicamera calibration package. When the intrinsic (focal length, optical center, and distortion) as well as extrinsic (relative rotation and translation) properties of a set of cameras are known, synchronized frames from those cameras can be used to triangulate landmarks indentified across their multiple points of view. With more cameras, this 3D tracking becomes more robust to occlusion and the inevitable errors in 2D landmark tracking and camera property estimates.

While OpenCV provides straightforward functions for the estimation of single camera intrinsic properties as well as estimates of the extrinsic properties of two cameras, there is no straightforward way to estimate extrinsic properties for more than two cameras. Performing this requires [bundle adjustment](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html), which demands an extensive series of computational steps and intermediate data tracking.

Caliscope automates this more complex calibration function along with providing visual feedback regarding parameter estimates at each stage of processing. Additionally, there are sample implementations of a Tracker class using Google's Mediapipe that demonstrate the capacity to integrate the full calibration results with landmark tracking tools to achieve 3D pose estimation. While Mediapipe pose estimation has limitations regarding accuracy and precision, it demonstrates a data processing pipeline that can easily integrate more powerful tracking tools as they emerge.

Please see our [docs](https://mprib.github.io/caliscope/) for details about installation, project setup, and general workflow.

---
### Demo Output

https://github.com/mprib/caliscope/assets/31831778/803a4ce8-4012-4da1-87b9-66c5e6b31c59

*The above was created using Caliscope, a 7 year old t440p laptop, and 4 webcams (~$25 each). This includes camera calibration, recording of synchronized frames (720p @ 24 fps), landmark detection, and point triangulation. Note that the webcam recording functionality is not in the current version, though will be restored in the future. Animated rig creation was done using an early stage Blender add-on project called [Rigmarole](https://github.com/mprib/rigmarole)*

---

## Quick Start

Please note that given the size of some core dependencies (OpenCV, Mediapipe, and PySide6 are among them) installation and initial launch can take a while. 

### Basic Installation and Launch
#### Windows

```bash
# Open Command Prompt and navigate to directory that will hold venv
# this does not need to be the same as where your project workspace is held
cd path\to\your\project

# Create a virtual environment named 'env' using Python 3.10
"C:\Path\To\Python3.10\python.exe" -m venv .venv

# Activate the virtual environment
.\env\Scripts\activate

# Your virtual environment is now active.
# You can install using pip
pip install caliscope

# Launch from the command line
caliscope
```

#### MacOS/Linux
```bash
# Open Command Prompt and navigate to directory that will hold venv
# this does not need to be the same as where your project workspace is held
cd path/to/your/project

# Create a virtual environment named 'venv' using Python 3.10
/path/to/python3.10 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Your virtual environment is now active.
# You can install using pip
pip3 install caliscope

# Launch from the command line
caliscope
```

### Basic Steps

1. Once the GUI launches, navigate to File->New/Open Project and create a folder to hold your project
  - A basic [project structure](https://mprib.github.io/caliscope/project_setup/) will be created here
2. Define a Charuco calibration board via the Charuco tab and print it out, fixing it to something flat
3. Record footage for the calibration according to the guidelines for the [intrinsic](https://mprib.github.io/caliscope/intrinsic_calibration/) and [extrinsic](https://mprib.github.io/caliscope/extrinsic_calibration/) calibrations.
4. Record synchronized motion capture trials
  - A companion project ([multiwebcam](https://github.com/mprib/multiwebcam)) has been set up to facilitate this though is still in early stages
5. Store video files within the project folder and reload the workspace
6. Run autocalibration on all cameras within the Cameras tab
7. Run "Calibration Capture Volume" from the Workspace Tab
8. Set origin within the capture volume (optional but helpful)
9. Run post-processing on individual recordings to generate 3D trajectory output

## Key Features

### Calibration board creation
- Easy creation of `png` files for ChArUco calibration boards 
- board definition can be changed across intrinsic and extrinsic calibration allowing greater flexibiltiy

### Intrinsic Camera Calibration
- Automated calculation of camera intrinsic properties from input video
  - Optical Center
  - Focal Length
  - Lens Distortion

- Visualization of distortion model to ensure reasonableness

### Extrinsic Camera Calibration
- Automated bundle adjustment to estimate 6 DoF relative position of cameras
- Visualizer to inspect the estimates from the bundle adjustment
- Setting of the World Origin within the visualizer to simplify data processing


### 3D Tracking
- Tracker API for integrating alternate tracking methods
  - 3 sample implementations with Google Mediapipe (Hands/Pose/Holistic)
- Automated application of landmark tracking to synchronized videos
- Triangulation of 3D landmark position based on calibrated cameras
- Gap-filling and butterworth filtering to smooth trajectory estimates

### Trajectory Output

- output to `.trc` file format for use in biomechanical modelling
- output to tidy `.csv` format with well-labelled headers for straightforward integration with other workflows
- companion project [Rigmarole](https://github.com/mprib/rigmarole) in development to facilitate creation of animated rigs in Blender

## Limitations

### Requires Frame Sync
The workflow currently requires you to provide your own synchronized frames or to provide [a file](project_setup.md#frame_time_historycsv) that specifies the time at which each frame was read so that caliscope can perform the synchronization itself. There are plans to manage this synchronization automatically through audio files, though that has not yet been implemented.

### Currently only using Mediapipe

Google's Mediapipe provides a relatively easy and efficient method for human subject tracking, though for many uses it is limiting. Caliscope has a general Tracker base class that is implemented in a few versions (Pose/Hands/Holistic). This has provided a proof of concept implementation of markerless tracking, though for more robust use the roadmap calls for integration with more powerful tools such as [MMPose](https://github.com/open-mmlab/mmpose) and [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut).

## Reporting Issues and Requesting Features

To report a bug or request a feature, please [open an issue](https://github.com/mprib/caliscope/issues). Please keep in mind that this is an open-source project supported by volunteer effort, so your patience is appreciated.

## General Questions and Conversation

Post any questions in the [Discussions](https://github.com/mprib/caliscope/discussions) section of the repo. 


## Acknowledgments

This project was inspired by [FreeMoCap](https://github.com/freemocap/freemocap) (FMC), which is spearheaded by [Jon Matthis, PhD](https://jonmatthis.com/) of the HuMoN Research Lab. The FMC calibration and triangulation system is built upon [Anipose](https://github.com/lambdaloop/anipose), created by Lili Karushchek, PhD. Caliscope was originally envisioned as an alternative calibration tool to Anipose that would allow more granular estimation of intrinsics as well as visual feedback during the calibration process. Several lines of of the original Anipose triangulation code are used in this code base, though otherwise it was written from the ground up. I'm grateful to Dr. Matthis for his time developing FreeMoCap, discussing it with me, pointing out important code considerations, and providing a great deal of information regarding open-source project management.

## License

Caliscope is licensed under the permissive [BSD 2-Clause license](https://opensource.org/license/bsd-2-clause/). The triangulation function was adapted from the [Anipose](https://github.com/lambdaloop/anipose) code base which is also licensed under the BSD-2 Clause. A primary dependency of this project is PySide6 which provides the GUI front end. PySide6 is licensed under the [LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.html). Caliscope does not modify the underlying source code of PySide6 which is available via [PyPI](https://pypi.org/project/PySide6/).
