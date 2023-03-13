
# Introduction

pyxyfy is a python package for converting 2D (x,y) point data from multiple cameras into 3D position estimates. The core calibration function is built on top of OpenCV's `calibrateCamera` and `stereoCalibrate` methods, with additional refinement of estimated camera positions using [bundle adjustment via SciPy](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html). 3D point positions can then be estimated with time synchronized 2D data.

This project was inspired by Anipose and seeks to provide similar functionality with improved visual feedback to the user. The calibration process is intended to be presented in more granular steps, allowing easier diagnosis of poor calibrations.
