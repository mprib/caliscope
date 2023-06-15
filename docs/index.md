<div align="center"><img src = "images/pyxy3d_logo.svg" width = "150"></div>



# Welcome


Pyxy3D is an open-source python package designed for two primary purposes: *calibrating* DIY motion capture systems and *triangulating* the 3D position  of tracked landmarks where (x,y) coordinates have been identified across concurrent frames. It includes a pre-built implementation of Google's Holistic Mediapipe, but aims to be tracker-agnostic and provides an API for implementing alternative tracking packages. Pyxy3D also provides functionality for recording synchronized frames from a small number of connected webcams. This may be sufficient for small-scale use cases, but it will become unstable as the resolution, frame rate, and number of cameras increases. Consequently, where larger capture volumes or greater spatial or temporal resolution is needed, alternate data capture techniques will be required. 


This project is at a very early stage so please bear with us while going through the inevitable growing pains that are ahead. You feedback is appreciated. If you have specific recommendations, please consider creating an [issue](https://github.com/mprib/pyxy3d/issues). If you have more general questions or thoughts about the project, please open up a thread in the [discussions](https://github.com/mprib/pyxy3d/discussions).

If you are just starting out here and trying to get a basic handle on what this is, what it does, and how it is used, it might be best to dive into the [FAQ](), or just watch this quick demo of a calibration and capture session:

<div align="center">
<iframe width="1280" height="720" src="https://www.youtube.com/embed/QHQKkLCE0e4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe> </div>

From there, the [installation]() guide will walk you through the process of getting a system up and running, the [calibration]() guide will help you get it dialed in, and the [motion capture]() guide will walk through the process of collecting data.

Please note that landmark position data is currently only being exported to `csv` and `trc` (used by biomechanists) formats, which very likely is not going to satisfy the needs of animators. If you have experience in character rigging and python programming and are interested in contributing, please reach out in [discussions]() to get the ball rolling on a game plan. Pull requests are welcome!