
# Introduction

Pyxyfy (*pixie-fi*) is a python package for converting 2D (x,y) point data from multiple cameras into 3D position estimates. The core calibration is built on top of [OpenCV](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) with additional refinement via bundle adjustment using [SciPy](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html). 3D point positions can then be estimated with time synchronized 2D data.

This project was inspired by [Anipose](https://www.sciencedirect.com/science/article/pii/S2211124721011797https://www.sciencedirect.com/science/article/pii/S2211124721011797) and seeks to provide similar functionality with improved visual feedback to the user. The calibration process is intended to be presented in more granular steps, with a particular emphasis on accurate estimation of camera intrinsics. Stereocalibration is used to initialize estimates of relative camera positions allowing for rapid convergence during bundle adjustment.
