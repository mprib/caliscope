

<div align="center"><img src = "pyxy3d/gui/icons/pyxy_logo.svg" width = "150"></div>


[Introduction](#introduction)

[Quick Start](#quick-start)

[Key Features](#key-features)

[Limitations](#limitations)

[Known Issues](#known-issues)
---
## About

Pyxy3D (*pixie-3d*) is an open-source **Py**thon package for converting 2D **(x,y)** point data to **3D** estimates. It is intended to serve as the calibration and triangulation workhorse of a low-cost DIY motion capture studio. It's core functionality includes: 

- the estimation of intrinsic (focal length/optical center/distortion) and extrinsic (rotation and translation) camera parameters via a GUI
- API for slotting various tracking solutions into the data pipeline
- triangulation of tracked points

The package comes included with a sample tracker using Google's Mediapipe which illustrates how to use the tracker API. The camera management backend allows for recording of synchronized frames from connected webcams, though the frame rate/resolution/number of cameras will be limited by the bandwidth of the current system.

![Quick_Demo](https://github.com/mprib/pyxy3d/assets/31831778/5fc8e15e-ca64-447b-86b8-69c64601199c)

## Quick Start

This package has only been successfully tested on Windows 10 and MacOS 12 Ventura. Limited testing on Linx (Ubuntu) has failed due to issues loading PyQt6.

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
python3.10 -m venv .venv
```
4. Activate the environment
```powershell
.\.venv\Scripts\activate
```

5. Install Pyxy3D
```powershell
pip install pyxy3d
```
Note that this will also install dependencies into the virtual environment, some of which are large (OpenCV, SciPy and Numpy are among the core dependencies). Complete download and installation may take several minutes. 

6. Launch Pyxy3D    
```powershell
pyxy3d
```


At this point, an application window should launch, though be aware that it may take several seconds for this to load. 
Refer to the [Quick Start Video Walkthrough](https://youtu.be/QHQKkLCE0e4) to see how to calibrate, record and process data

## Key Features

The project leans heavily upon OpenCV, SciPy, and PyQt to provide the following **key features**:

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

## Limitations

Please note that the system currently has the following **limitations**:
- MediaPipe is only configured to run on Windows
    - while the camera calibration will likely work on other systems, the included sample markerless tracking will not (currently)
- It does not support anything other than standard webcams at the moment 
- The frame capture backend presents a primary bottleneck that will limit the number of cameras/resolution/frame rates that can be used, which ultimately limits the size and precision of the capture volume.
- Data export is currently limited to .csv, and .trc files. Use in 3D animation tools like Blender, which require character rigging, will require additional processing.

## Known Issues
### Seg Faults on Windows 10
[June 23, 2023] The most recent version of the package on PyPI (0.1.3) has been updated to allow it to run on MacOS 12. This update did not change any code, but did change which versions of the underlying dependencies where configured to install. Segmentation faults during recording emerged after this update and are currently being addressed as a top priority.

## Reporting Issues and Requesting Features

To report a bug or request a feature, please [open an issue](https://github.com/mprib/pyxy3d/issues). Please keep in mind that this is an open-source project supported by volunteer effort, so your patience is appreciated.

## General Questions and Conversation

Post any questions in the [Discussions](https://github.com/mprib/pyxy3d/discussions) section of the repo. 

## Acknowledgments

This project was inspired by [FreeMoCap](https://github.com/freemocap/freemocap) (FMC), which is spearheaded by [Jon Matthis, PhD](https://jonmatthis.com/) of the HuMoN Research Lab. The FMC calibration and triangulation system is built upon [Anipose](https://github.com/lambdaloop/anipose), created by Lili Karushchek, PhD. Several lines of FMC/Anipose code are used in the triangulation methods of Pyxy3D.  I'm grateful to Dr. Matthis for his time developing FreeMoCap, discussing it with me, pointing out important code considerations, and providing a great deal of information regarding open-source project management.

I began my python programming journey in August 2022. Hoping to understand the Anipose code, I started learning the basics of OpenCV. [Murtaza Hassan's](https://www.youtube.com/watch?v=01sAkU_NvOY) computer vision course rapidly got me up to speed on performing basic frame reading and parsing of Mediapipe data. To get a grounding in the fundamentals of camera calibration and triangulation I followed the excellent blog posts of [Temuge Batpurev](https://temugeb.github.io/). At the conclusion of those tutorials I decided to try to "roll my own" calibration and triangulation system as a learning exercise (which slowly turned into this repository). Videos from [GetIntoGameDev](https://www.youtube.com/watch?v=nCWApy9gCQQ) helped me through projection transforms. The excellent YouTube lectures of [Cyrill Stachniss](https://www.youtube.com/watch?v=sobyKHwgB0Y) provided a foundation for understanding the bundle adjustment process, and the [SciPy Cookbook](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html) held my hand when implementing the code for this optimization. Debugging the daisy-chain approach to parameter initialization would not have been possible without the highly usable 3D visualization features of [PyQtGraph](https://www.pyqtgraph.org/).

[ArjanCodes](https://www.youtube.com/@ArjanCodes) has been an excellent resource for Python knowledge, as has [Corey Schafer](https://www.youtube.com/channel/UCCezIgC97PvUuR4_gbFUs5g) whose videos on multithreading were invaluable at tackling early technical hurdles related to concurrent frame reading. 

While Pyxy3D is not a fork of any pre-existing project, it would not exist without the considerable previous work of many people, and I'm grateful to them all.

## License

Pyxy3D is licensed under AGPL-3.0.
