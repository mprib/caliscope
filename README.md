# A Branch for Learning Things and Slowly Rolling My Own FMC

## PYQT6 Mock Up

Implement an OpenCV camera calibration with a PyQT6 user interface. I will begin by following the tutorial here

### Initial tutorial

https://www.youtube.com/watch?v=s72xCnaidso

NOTES: I have gone through some of this to get a general sense of the structure of the pyqt6 code. I think I have a rough handle on it, and am moving forward to issues of concurrency to read frames at a known framerate at the same time, and then expose them to real-time processing. This is the backbone of any system that wants to do what FMC is doing.

## Concurrency

This tutorial was linked do on a github issue:

https://realpython.com/python-concurrency/

I would like to start getting more in the weeds on camera I/O and multiprocessing of frames with a controlled framerate. This is a cool challenge, and I think that if you get a good, clean solution to this, then you can extend it to calibration and real-time frame processing.

It appears that I'm able to read in multiple frames here with the camera widget. Now I would like to update each frame with the framerate so that I have an understanding of what that is. I previously saw a mediapipe demo on youtube that did this, and I'll go to that to get an idea of how to proceed with this.