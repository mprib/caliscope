

<div align="center"><img src = "pyxy3d/gui/icons/pyxy_logo.svg" width = "150"></div>

## Description

Pyxy3D is an open-source python tool for converting two-dimensional (x,y) coordinates obtained from multiple standard webcams into 3D point estimates. It provides an integrated system for camera calibration and point triangulation that enables the creation of cost-efficient small scale motion capture systems. When combined with markerless tracking algorithms such as Google's Mediapipe, it is possible to perform markerless 3D tracking with a standard computer and a couple webcams. 

## Key Features

The project leans heavily upon OpenCV, SciPy, and PyQt to provide the following **key features**:

- User-friendly GUI
- Easy design and creation of charuco calibration board
- Both extrinsic and intrinsic camera parameters are estimated
- optional double-sided charuco board for better positional estimates of cameras placed opposite each other
- Visual feedback during the calibration process including corner tracking, distortion modelling, and camera position estimates
- World origin setting using the calibration board 
- Fast convergence during bundle adjustment due to parameter initializations based on daisy-chained stereopairs of cameras
- Recording of synchronized frames from connected webcams for post-processing
- Tracker API for future extensibility with included sample implementation using Mediapipe 
- Triangulation of tracked landmarks
- Visualization of triangulated points for quick confirmation of output quality
- currently exporting to `.csv` and `.trc` file formats

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



## Installation

Pyxy3D is installable via pip and the GUI can be launched from the command line. It is **strongly** advised that you do so within a virtual environment. The package requires [Python 3.10](https://www.python.org/downloads/release/python-3100/)  or higher. Because the Mediapipe implementation only works on Windows currently, these steps assume you are installing on Windows 10.

### 1. Create a virtual environment

Find the path to your python.exe file. You can install Python 3.10 from [here](https://www.python.org/downloads/release/python-3100/). For me the path is `C:\Python310\python.exe`

Create a folder where you would like the code and virtual environment to live. This can be different from the folder where your motion capture calibration and recording data is stored. Open the folder and right click within it, select  ![[Pasted image 20230608102647.png]] from the context menu to launch a terminal. 
   
Run the following at the command prompt. Substitute in the path to `python.exe` that is true for your machine
```
C:\Python310\python.exe -m venv .venv
```

This will create a fresh version of python within that folder which you will use to manage your project. Activate the environment using the following command (if this exact command doesn't work, then [some other variation will](https://docs.python.org/3/library/venv.html#how-venvs-work))
```
.\.venv\Scripts\activate
```

The terminal should now show the environment is activated with something like this green parenthetical:
![[Pasted image 20230608095719.png]]

You can confirm that your python path is set by running

```
python -c "import sys; print(sys.executable)"
```
which should point to the file in the virtual environment you created:
![[Pasted image 20230608100059.png]]

### 2. Install pyxy3D via pip

For extra caution, upgrade pip with the following command:
```
pip install --upgrade pip
```

You are now ready to install pyxy3D from the Python Package Index (PyPI) via pip:

```
pip install pyxy3d
```

Installation may take a moment...

### 3. Launch from the command line

With the package installed and the virtual environment activated, run the following command to launch the tool:

```
pyxy3d
```

A window should pop up and you can proceed with the capture session

### Example session

For a detailed walkthrough, refer to the project's comprehensive [documentation](link_to_documentation).

## Reporting Issues and Requesting Features

To report a bug or request a feature, please open an issue.

## General Questions and Conversation

Post any questions in the Discussions section of the repo. As you post your questions, please keep in mind that this is an open-source project supported by volunteer effort, so your patience is appreciated.

## Acknowledgments

This project was inspired by [FreeMoCap](https://github.com/freemocap/freemocap) (FMC), which is spearheaded by [Jon Matthis, PhD](https://jonmatthis.com/) of the HuMoN Research Lab. The FMC calibration and triangulation system is built upon [Anipose](https://github.com/lambdaloop/anipose), created by Lili Karushchek, PhD. Several lines of FMC/Anipose code are used in the triangulation methods of Pyxy3D. I'm grateful to Dr. Matthis' for his time developing FreeMoCap, discussing it with me, pointing out important code considerations, and providing a great deal of information regarding open-source project management.

I began my python programming journey in August 2022. Hoping to understand the Anipose code, I started learning the basics of OpenCV. [Murtaza Hassan's](https://www.youtube.com/watch?v=01sAkU_NvOY) computer vision course rapidly got me up to speed on performing basic frame reading and parsing of Mediapipe data. To get a grounding in the fundamentals of camera calibration and triangulation I followed the excellent blog posts of [Temuge Batpurev](https://temugeb.github.io/). At the conclusion of those tutorials I decided to try to "roll my own" calibration and triangulation system as a learning exercise (which slowly turned into this repository). Videos from [GetIntoGameDev](https://www.youtube.com/watch?v=nCWApy9gCQQ) helped me through some dark and confusing times when this Physical Therapist tried to wrap his head around projection transforms. The excellent YouTube lectures of [Cyrill Stachniss](https://www.youtube.com/watch?v=sobyKHwgB0Y) provided a foundation for understanding the bundle adjustment process, and the [SciPy Cookbook](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html) held my hand when implementing the code for this optimization. Debugging the daisy-chain approach to parameter initialization would not have been possible without the highly usable 3D visualization features of [PyQtGraph](https://www.pyqtgraph.org/).

[ArjanCodes](https://www.youtube.com/@ArjanCodes) has been a frequent touchstone in my coding journey and is an incredible source of Python knowledge, as is [Corey Schafer](https://www.youtube.com/channel/UCCezIgC97PvUuR4_gbFUs5g) whose videos on multithreading were invaluable at tackling early technical hurdles related to concurrent frame reading. 

While Pyxy3D is not a fork of any pre-existing project, it would not exist without the considerable previous work of many people, and I'm grateful to them all.

## License

Pyxy3D is licensed under AGPL-3.0.
