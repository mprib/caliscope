<div align="center"><img src = "pyxy3d/gui/icons/pyxy_logo.svg" width = "150"></div>

<div align="center">
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pyxy3d?color=blue)]()
[![PyPI - License](https://img.shields.io/pypi/l/pyxy3d?color=blue)]()
[![PyPI - Version](https://img.shields.io/pypi/v/pyxy3d?color=blue)]()
</div>


## About

Pyxy3d (*pixie-3d*) is a **py**thon package that integrates:

- multicamera calibration
- 2D (**x,y**) landmark tracking
- **3D** landmark triangulation. 

It is GUI-based, permissively licensed under the LGPLv3, and intended to serve as the processing hub of a low-cost DIY motion capture studio.

Currently it uses Google's Mediapipe for markerless tracking, though is built on a Tracker API that can be used to integrate alternate landmark tracking tools.

Please see our [docs](https://mprib.github.io/pyxy3d/) for details about installation, project setup, and general workflow.

---
### Demo Output
https://github.com/mprib/pyxy3d/assets/31831778/803a4ce8-4012-4da1-87b9-66c5e6b31c59

*The above was created using Pyxy3D, a 7 year old t440p laptop, and 4 webcams (~$25 each). This includes camera calibration, recording of synchronized frames (720p @ 24 fps), landmark detection, and point triangulation. Note that the webcam recording functionality is not in the current version, though will be restored in the future. Animated rig creation was done using an early stage Blender add-on project called [Rigmarole](https://github.com/mprib/rigmarole)*

---

## Quick Start

Please note that given the size of some core dependencies (OpenCV, Mediapipe, and PySide6 are among them) installation and initial launch can take a while. 

### Windows

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

### MacOS/Linux
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
