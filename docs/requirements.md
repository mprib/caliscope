# Minimum Requirements

- **Python version**: [Python 3.10](https://www.python.org/downloads/release/python-3100/) or [3.11](https://www.python.org/downloads/release/python-3110/)

- **Operating System**: While the core calibration and GUI are cross-platform, Google's [MediaPipe](https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md) is currently used for pose tracking and has only been tested on Windows 10 and MacOS.

- **Cameras**: At least two cameras are needed, though tracking accuracy improves with the use of more cameras. Mediapipe tracking is in-the-box 

- **Computer**: 

- **Calibration Board**: A [ChArUco board](https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html) is needed for camera calibration and determining spatial relationships between multiple cameras. A sample board can be printed from the GUI on a standard 8.5 x 11 sheet of paper, though . It is crucial to place the printed board on a flat surface to ensure accurate calibration, such as taping it down to a rigid flat piece of cardboard.

- **Capture Environment**: Data recording requires a well-lit and evenly lit environment. It is beneficial if the background contrasts highly with the individual being tracked. For example, tracking difficulties may arise if a person in dark clothing stands against a similarly dark wall. Be mindful of clothing, background, and lighting to optimize the quality of the captured data.