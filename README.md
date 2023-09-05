

<div align="center"><img src = "pyxy3d/gui/icons/pyxy_logo.svg" width = "150"></div>

<div align="center">

[Quick Start](#quick-start) | [Key Features](#key-features) | [Limitations](#limitations)
</div>


https://github.com/mprib/pyxy3d/assets/31831778/96413cf8-52d7-4054-ab13-a5f8daa4cbf9

The above was created using Pyxy3D, a t440p laptop, and 4 webcams (~$25 each). This includes camera calibration, recording of synchronized frames (720p @ 24 fps), landmark detection, and point triangulation. 

The Blender output process is currently a bit ad hoc. The plan is to evolve that workflow into a dedicated Blender add-on (also free and open source) that can process the output of pyxy3d and create rigged animations in blender. Thanks to Dorothy Overbey for being a test subject (and a wonderful wife).


<div align="center"><img src = "https://github.com/mprib/pyxy3d/assets/31831778/2f235ecd-0f1d-4bb8-9935-2ed21517cd0e" width = "50%"></div>

---
## About

Pyxy3D (*pixie-3d*) is an open-source motion capture tool that allows for markerless motion tracking with a typical computer (Windows or MacOS currently) and 2 or more webcams. In other words, it is a **Py**thon package for converting 2D **(x,y)** point data to **3D** estimates. It's core functionality includes: 

- the estimation of intrinsic (focal length/optical center/distortion) and extrinsic (rotation and translation) camera parameters via a GUI
- API for slotting various tracking solutions into the data pipeline (currently works with Google's Mediapipe)
- triangulation of tracked points


## Key Features

The project leans heavily upon OpenCV, SciPy, PySide6 and Google's Mediapipe to provide the following **key features**:

[Charuco Board Creation](#charuco-board-creation)


[Camera Configuration](#camera-configuration)


[Intrinsic Camera Calibration](#intrinsic-camera-calibration)


[Multicamera Calibration](#multicamera-calibration)


[Recording of Synchronized Frames](#record-synchronized-frames)


[Post-Processing](#post-processing)

---

### Charuco Board Creation

A .png file of the board is saved and the calibration tracking will refer to this board. A mirror image board can also be used to improve multicamera calibration data collection for cameras that may not have an easy view of a single plane of the board.

https://github.com/mprib/pyxy3d/assets/31831778/b8b4736f-c7d7-4ba4-a92a-6817ba9cbf61

---

### Camera Configuration
Connect to available webcams, set exposure, resolution, and adjust for any rotation in the camera.

https://github.com/mprib/pyxy3d/assets/31831778/2fe0181d-eaf0-4b76-84db-98bedfc9dbee

---

### Intrinsic Camera Calibration

With visual feedback to confirm quality of recorded data, intrinsic camera characteristics (camera matrix and distortion are estimated). In addition to displaying RMSE, the distortion model can be applied to the live video feed to assess reasonability. Once good estimates of these parameters are estimated for a given camera, they don't need to be estimated again, allowing quick multicamera recalibration as the recording setup changes.

https://github.com/mprib/pyxy3d/assets/31831778/b975546f-8ba1-481e-8fd1-29be5e565572

---

### Multicamera Calibration

Visual feedback, target board counts, and real-time checks on the collected data ensure that the bundle adjustment optimization quickly converges to a solution given good intrinisic camera parameter estimates and sensible initialization of 6DoF estimates based on daisy-chained stereopairs.

https://github.com/mprib/pyxy3d/assets/31831778/2a02b10f-b7a8-4dba-ac15-965605f42f6f

---

### Record Synchronized Frames

Frames are synchronized in real-time meaning that frame drops can be assessed and the target frame rate can be adjusted as need be. In the example below, port 1 drops 1-2% of frames at 30fps, but this resolves with a minor tweak to the frame rate.

https://github.com/mprib/pyxy3d/assets/31831778/08656444-e846-4dbc-b278-51f0ab8d76db

---

### Post-Processing

https://github.com/mprib/pyxy3d/assets/31831778/25bdf3a1-2bd0-48e4-a4d8-2815867c94ff

Recordings can be processed with built in landmark trackers which are currently based around Google's Mediapipe. The post-processing pipeline goes through several stages:

1. landmark identification across all recordings
2. small 2D gap-filling (<3 frames)
3. point triangulation
4. small 3d gap-filling (<3 frames)
5. trajectory smoothing (bidirectional butterworth at 6Hz)

Results are visualized in the pyqtgraph window for checking quality of results. Labelled (x,y,z) coordinates are saved in a `.csv` file accessible from the recording directory (can be opened from the post-processing tab). When  full body data is tracked (I'm at my desk in this walk-through so not applicable) a configuration file can be generated that specificies mean distances between landmarks. This configuration was used to auto-scale the metarig animation shown in the ballet video at the top.

---

## Quick Start

This package has only been successfully tested on Windows 10 and MacOS 12 Ventura. 

From a terminal (the code below is using Powershell), do the following:

1. Create a new project folder
```powershell
mkdir pyxy3d_demo
```
2. Navigate into that directory
```powershell
cd pyxy3d_demo
```
3. Create a virtual environment with [Python 3.10](https://www.python.org/downloads/release/python-3100/) or later:
```powershell
C:\Python310\python.exe -m venv .venv
```
4. Activate the environment
```powershell
.\.venv\Scripts\activate
```

5. Install Pyxy3D
```powershell
pip install pyxy3d
```
Note that this will also install dependencies into the virtual environment, some of which are large (OpenCV, SciPy, Numpy and Mediapipe are among the core dependencies). Complete download and installation may take several minutes. 

6. Launch Pyxy3D    
```powershell
pyxy3d
```


At this point, an application window will launch, though be aware that it may take several seconds for this to load particularly on the first launch on your machine. 
Refer to the [Quick Start Video Walkthrough](https://youtu.be/QHQKkLCE0e4) to see how to calibrate, record and process data



## Limitations

Please note that the system currently has the following **limitations**:
- It does not support anything other than standard webcams at the moment (a pipeline for processing pre-recorded videos is in the works).
- The frame capture backend presents a primary bottleneck that will limit the number of cameras/resolution/frame rates that can be used, which ultimately limits the size and precision of the capture volume.
- Data export is currently limited to .csv, and .trc files. Use in 3D animation tools like Blender, which require character rigging, will require additional processing.



## Reporting Issues and Requesting Features

To report a bug or request a feature, please [open an issue](https://github.com/mprib/pyxy3d/issues). Please keep in mind that this is an open-source project supported by volunteer effort, so your patience is appreciated.

## General Questions and Conversation

Post any questions in the [Discussions](https://github.com/mprib/pyxy3d/discussions) section of the repo. 


## Acknowledgments

This project was inspired by [FreeMoCap](https://github.com/freemocap/freemocap) (FMC), which is spearheaded by [Jon Matthis, PhD](https://jonmatthis.com/) of the HuMoN Research Lab. The FMC calibration and triangulation system is built upon [Anipose](https://github.com/lambdaloop/anipose), created by Lili Karushchek, PhD. Pyxy3D was originally envisioned as an alternative calibration tool to Anipose that would allow more granular estimation of intrinsics as well as visual feedback during the calibration process. Several lines of of the original Anipose triangulation code are used in this code base, though otherwise it was written from the ground up. I'm grateful to Dr. Matthis for his time developing FreeMoCap, discussing it with me, pointing out important code considerations, and providing a great deal of information regarding open-source project management.

## License

Pyxy3D is licensed under AGPL-3.0.
