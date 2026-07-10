# Sample Project

A sample dataset demonstrating the calibration workflow is available for [download](https://github.com/mprib/caliscope/releases/download/v0.8.0/lab_test_data.zip).

The sample project is a 5-camera setup with charuco extrinsic calibration.
The cameras were not hardware-synchronized, so the data includes a `timestamps.csv` for frame alignment.
It contains raw input data only: intrinsic calibration videos, extrinsic calibration videos, and one walking recording.
You configure calibration targets and run the pipeline yourself.

The intrinsic calibration videos use a chessboard rather than a charuco board. This produces a slightly worse intrinsic calibration (~1 px reprojection error). A charuco board will give better results. The extrinsic calibration uses a charuco board.

For a scripting-based alternative to the GUI walkthrough, see `scripts/demo_api.py` which runs the same calibration pipeline programmatically.
