<div align="center">

# Open Source DIY Motion Capture

<img src = "pyxy3d/gui/icons/pyxy_logo.svg" width = "150">

[![PyPI - Downloads](https://img.shields.io/pypi/dm/pyxy3d?color=blue)](https://pypi.org/project/pyxy3d/)
[![PyPI - License](https://img.shields.io/pypi/l/pyxy3d?color=blue)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![PyPI - Version](https://img.shields.io/pypi/v/pyxy3d?color=blue)](https://pypi.org/project/pyxy3d/)
[![GitHub last commit](https://img.shields.io/github/last-commit/mprib/pyxy3d.svg)](https://github.com/mprib/pyxy3d/commits)
[![GitHub stars](https://img.shields.io/github/stars/mprib/pyxy3d.svg?style=social&label=Star)](https://github.com/mprib/pyxy3d/stargazers)
</div>

## About
Pyxy3D (*pixie-3D*) is intended to serve as the core software component of a low-cost DIY motion capture studio. It is a **py**thon package that integrates:

- multicamera calibration
- 2D (**x,y**) landmark tracking
- **3D** landmark triangulation. 

It is GUI-based and open-sourced under the LGPLv3.

Landmark tracking is based on a Tracker abstract base class. Variations of Google's Mediapipe have been implemented to illustrate use of this base class and how these calculations will flow automatically through the processing pipeline. Implementing alternate tracking tools (such as MMPose and DeepLabCut) is on the development roadmap.

Please see our [docs](https://mprib.github.io/pyxy3d/) for details about installation, project setup, and general workflow.

---
### Demo Output
https://github.com/mprib/pyxy3d/assets/31831778/803a4ce8-4012-4da1-87b9-66c5e6b31c59

*The above was created using Pyxy3D, a 7 year old t440p laptop, and 4 webcams (~$25 each). This includes camera calibration, recording of synchronized frames (720p @ 24 fps), landmark detection, and point triangulation. Note that the webcam recording functionality is not in the current version, though will be restored in the future. Animated rig creation was done using an early stage Blender add-on project called [Rigmarole](https://github.com/mprib/rigmarole)*

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
pip install pyxy3d

# Launch from the command line
pyxy3d
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
pip3 install pyxy3d

# Launch from the command line
pyxy3d
```

### Basic Steps

1. Once the GUI launches, navigate to File->New/Open Project and create a folder to hold your project
  - A basic [project structure](https://mprib.github.io/pyxy3d/project_setup/) will be created here
2. Define a Charuco calibration board via the Charuco tab and print it out, fixing it to something flat
3. Record footage for the calibration according to the guidelines for the [intrinsic](https://mprib.github.io/pyxy3d/intrinsic_calibration/) and [extrinsic](https://mprib.github.io/pyxy3d/extrinsic_calibration/) calibrations.
4. Record synchronized motion capture trials
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
