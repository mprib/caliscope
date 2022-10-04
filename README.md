# A Branch for Learning Things and Slowly Rolling My Own FMC

## PYQT6 Mock Up

Implement an OpenCV camera calibration with a PyQT6 user interface. I will begin by following the tutorial here

### Initial tutorial

https://www.youtube.com/watch?v=s72xCnaidso

NOTES: I have gone through some of this to get a general sense of the structure of the pyqt6 code. I think I have a rough handle on it, and am moving forward to issues of concurrency to read frames at a known framerate at the same time, and then expose them to real-time processing. This is the backbone of any system that wants to do what FMC is doing.

---

I am returning today (10-3-2022) to PyQt6 and looking to get a better handle on how to create a GUI for the application that will allow a more streamlined camera placement, calibration, capture, and post-processing. My own code is becoming a bit too disparate for me to sensibly maintain it and I need to start aligning/organizing things in a central way.

## Concurrency

This tutorial was linked do on a github issue:

https://realpython.com/python-concurrency/

I would like to start getting more in the weeds on camera I/O and multiprocessing of frames with a controlled framerate. This is a cool challenge, and I think that if you get a good, clean solution to this, then you can extend it to calibration and real-time frame processing.

It appears that I'm able to read in multiple frames here with the camera widget. Now I would like to update each frame with the framerate so that I have an understanding of what that is. I previously saw a mediapipe demo on youtube that did this, and I'll go to that to get an idea of how to proceed with this.

Alright, so I'm now reading in multiple cameras using threading, and appear to be getting ~20 FPS from these $30 webcams. Now what?

I want to take a look at how I can incorporate some of the previous code I developed for the calibration. Perhaps just to expand the footprint of my knowledge for a bit, I want to see if I can get the mediapipe hand detection working here as well. That may provide some insight into the challenges of more intense frame processing.