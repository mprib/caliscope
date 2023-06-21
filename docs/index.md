<div align="center"><img src = "images/pyxy3d_logo.svg" width = "150"></div>



# Welcome

Pyxy3D (*pixie-3d*) is an open-source **Py**thon package for converting 2D **(x,y)** point data to **3D** estimates. It is intended to serve as the calibration and triangulation workhorse of a low-cost DIY motion capture studio. It's core functionality includes: 

- the estimation of intrinsic (focal length/optical center/distortion) and extrinsic (rotation and translation) camera parameters via a GUI
- API for slotting various tracking solutions into the data pipeline
- triangulation of tracked points

The packages comes included with a sample tracker using Google's Mediapipe which illustrates how to use the tracker API. The camera management backend allows for recording of synchronized frames from connected webcams, though the frame rate/resolution/number of cameras will be limited by the bandwidth of the current system.

This project is at a very early stage so please bear with us while going through the inevitable growing pains that are ahead. You feedback is appreciated. If you have specific recommendations, please consider creating an [issue](https://github.com/mprib/pyxy3d/issues). If you have more general questions or thoughts about the project, please open up a thread in the [discussions](https://github.com/mprib/pyxy3d/discussions).

If you are just starting out here and trying to get a basic handle on what this is, what it does, and how it is used, it might be best to dive into the [FAQ](), or just watch this quick demo of a calibration and capture session:

<div align="center">
<iframe width="1280" height="720" src="https://www.youtube.com/embed/QHQKkLCE0e4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe> </div>

From there, the [installation]() guide will walk you through the process of getting a system up and running, the [calibration]() guide will help you get it dialed in, and the [motion capture]() guide will walk through the process of collecting data.

Please note that landmark position data is currently only being exported to `csv` and `trc` (used by biomechanists) formats, which very likely is not going to satisfy the needs of animators. If you have experience in character rigging and python programming and are interested in contributing, please reach out in [discussions]() to get the ball rolling on a game plan. Pull requests are welcome!