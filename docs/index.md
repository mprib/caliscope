<div align="center"><img src = "images/pyxy3d_logo.svg" width = "150"></div>


# Welcome

Pyxy3d (*pixie-3d*) is a **py**thon package for multicamera calibration  that integrates with 2D (**x,y**) landmark tracking to triangulate **3D** landmark positions. It is GUI-based, permissively licensed under the LGPLv3, and intended to serve as the processing hub of a low-cost DIY motion capture studio. It's core functionality includes: 

- the estimation of intrinsic (focal length/optical center/distortion) and extrinsic (rotation and translation) camera parameters
- API for slotting various tracking solutions into the data pipeline
- triangulation of tracked points

The packages comes included with a sample tracker using Google's Mediapipe which illustrates how to use the tracker API. 

The workflow is currently BYOSF (Bring Your Own Synchronized Frames). Pyxy3d will take care of the rest.

The [installation]() guide will walk you through the process of installing the package on your system, [requirements] will getting a system up and running, the [calibration]() guide will help you get it dialed in, and the [motion capture]() guide will walk through the process of collecting data.

This project is at a very early stage so please bear with us while going through the inevitable growing pains that are ahead. You feedback is appreciated. If you have specific recommendations, please consider creating an [issue](https://github.com/mprib/pyxy3d/issues). If you have more general questions or thoughts about the project, please open up a thread in the [discussions](https://github.com/mprib/pyxy3d/discussions).

<div align="center">
<iframe width="1280" height="720" src="https://www.youtube.com/embed/QHQKkLCE0e4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe> </div>

