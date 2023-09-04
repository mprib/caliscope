

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

The project leans heavily upon OpenCV, SciPy, and PySide6 to provide the following **key features**:

- User-friendly graphical user interface (GUI)
- Easy creation and modification of the charuco calibration board
- Both extrinsic and intrinsic camera parameters are estimated
- Optional double-sided charuco board for better positional estimates of cameras placed opposite each other
- Visual feedback during the calibration process 
- World origin setting using the calibration board 
- Fast convergence during bundle adjustment due to parameter initializations based on daisy-chained stereopairs of cameras
- Recording of synchronized frames from connected webcams for post-processing
- Tracker API for future extensibility with included sample implementation using Mediapipe 
- Triangulation of tracked landmarks
- Visualization of triangulated points for quick confirmation of output quality
- Currently exporting to `.csv` and `.trc` file formats

| Feature | Demo|
|---------|-----|
|Charuco Board .png creation|https://github.com/mprib/pyxy3d/assets/31831778/1fde65a9-3386-40e1-95c2-3bc05ea71a03|




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
