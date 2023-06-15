

<div align="center"><img src = "pyxy3d/gui/icons/pyxy_logo.svg" width = "150"></div>

#### Table of Contents

[Introduction](#introduction)

[Quick Start](#quick-start)

[Key Features](#key-features)

[Limitations](#limitations)

[Known Issues](#known-issues)

---
## About
Pyxy3D is an open-source python package designed for two primary purposes: calibrating DIY motion capture systems and triangulating the 3D position of tracked landmarks where (x,y) coordinates have been identified across concurrent frames. 

It includes a pre-built implementation of Google's Holistic Mediapipe, but aims to be tracker-agnostic and provides an API for implementing alternative tracking packages. Pyxy3D also provides functionality for recording synchronized frames from a small number of connected webcams. This may be sufficient for small-scale use cases, but it will become unstable as the resolution, frame rate, and number of cameras increases. Consequently, where larger capture volumes or greater spatial or temporal resolution is needed, alternate data capture techniques will be required. 
Pyxy3D is an open-source python tool for converting two-dimensional (x,y) coordinates obtained from multiple standard webcams into 3D point estimates. It provides an integrated system for camera calibration and point triangulation that enables the creation of cost-efficient small scale motion capture systems. When combined with markerless tracking algorithms such as Google's Mediapipe, it is possible to perform markerless 3D tracking with a standard computer and a couple webcams. 

## Quick Start

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
    - while the camera calibration will likely work on other systems, the markerless tracking will not (currently)
- It does not support anything other than standard webcams. 
    - I currently have no intention of supporting mobile phones as cameras for the system
- Based on recent testing, some webcams will deliver poor connection times/frame rates/calibrations. I'm currently using 4 EMEET SmartCam C960 cameras. These are inexpensive (~$25 each) and readily available. They deliver decent results at 30 fps and 720p. I welcome feedback about user experiences with other cameras
- No real-time tracking
    - the underlying data processing pipeline was designed to accommodate real-time tracking but I want to make sure that everything works well with the simpler and more stable post-processing workflow before trying to get that implemented in an integrated way
- Data export is currently limited to .csv, and .trc files. Use in 3D animation tools like Blender, which require character rigging, will require additional processing.

## Known Issues

The main GUI allows for accessing of all of the package's functionality at once, though this imposes some additional processing overhead that can undermine recording, and switching between GUI modes can provoke crashes. Improvements are on the To Do list, but in the meantime can be sidestepped by launching individual widgets from the command line as [described below](#3-launch-from-the-command-line)



## Reporting Issues and Requesting Features

To report a bug or request a feature, please [open an issue](https://github.com/mprib/pyxy3d/issues). Please keep in mind that this is an open-source project supported by volunteer effort, so your patience is appreciated.

## General Questions and Conversation

Post any questions in the [Discussions](https://github.com/mprib/pyxy3d/discussions) section of the repo. 

## Acknowledgments

This project was inspired by [FreeMoCap](https://github.com/freemocap/freemocap) (FMC), which is spearheaded by [Jon Matthis, PhD](https://jonmatthis.com/) of the HuMoN Research Lab. The FMC calibration and triangulation system is built upon [Anipose](https://github.com/lambdaloop/anipose), created by Lili Karushchek, PhD. Several lines of FMC/Anipose code are used in the triangulation methods of Pyxy3D. Pyxy3D is my attempt at helping to move toward an open source tool for motion capture that can hopefully one day benefit scientists, clinicians, and artists alike. I'm grateful to Dr. Matthis for his time developing FreeMoCap, discussing it with me, pointing out important code considerations, and providing a great deal of information regarding open-source project management.

I began my python programming journey in August 2022. Hoping to understand the Anipose code, I started learning the basics of OpenCV. [Murtaza Hassan's](https://www.youtube.com/watch?v=01sAkU_NvOY) computer vision course rapidly got me up to speed on performing basic frame reading and parsing of Mediapipe data. To get a grounding in the fundamentals of camera calibration and triangulation I followed the excellent blog posts of [Temuge Batpurev](https://temugeb.github.io/). At the conclusion of those tutorials I decided to try to "roll my own" calibration and triangulation system as a learning exercise (which slowly turned into this repository). Videos from [GetIntoGameDev](https://www.youtube.com/watch?v=nCWApy9gCQQ) helped me through projection transforms. The excellent YouTube lectures of [Cyrill Stachniss](https://www.youtube.com/watch?v=sobyKHwgB0Y) provided a foundation for understanding the bundle adjustment process, and the [SciPy Cookbook](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html) held my hand when implementing the code for this optimization. Debugging the daisy-chain approach to parameter initialization would not have been possible without the highly usable 3D visualization features of [PyQtGraph](https://www.pyqtgraph.org/).

[ArjanCodes](https://www.youtube.com/@ArjanCodes) has been an excellent resource for Python knowledge, as has [Corey Schafer](https://www.youtube.com/channel/UCCezIgC97PvUuR4_gbFUs5g) whose videos on multithreading were invaluable at tackling early technical hurdles related to concurrent frame reading. 

While Pyxy3D is not a fork of any pre-existing project, it would not exist without the considerable previous work of many people, and I'm grateful to them all.

## License

Pyxy3D is licensed under AGPL-3.0.
