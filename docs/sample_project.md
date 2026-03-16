# Sample Project

A sample dataset demonstrating the full calibration and reconstruction workflow is available for [download](https://1drv.ms/f/c/a30b139c66ff49c7/EqIVjIRLQ9hEh7hLE7UysAcBvxa1Oqy8JlM8Cu1gg0mXKw?e=3PAYXa).

The sample project is a 3-camera setup with ArUco marker extrinsic calibration and software synchronization. It contains raw input data only: intrinsic calibration videos, extrinsic calibration videos with timestamps, and one walking recording. You configure calibration targets and run the pipeline yourself following the documentation.

This project illustrates the workflow with a minimal setup. Several improvements would increase the quality of the final results:

- more cameras
- higher resolution and frame rate
- better lighting
- larger calibration board
- hardware synchronization rather than software frame alignment

For a scripting-based alternative to the GUI walkthrough, see `scripts/demo_api.py` which exercises the same calibration pipeline programmatically.
