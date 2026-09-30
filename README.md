<div align="center">

# Caliscope

*Multicamera calibration for 3D motion capture*

[![PyPI - Downloads](https://img.shields.io/pypi/dm/caliscope?color=blue)](https://pypi.org/project/caliscope/)
[![PyPI - License](https://img.shields.io/pypi/l/caliscope?color=blue)](https://opensource.org/license/bsd-2-clause/)
[![PyPI - Version](https://img.shields.io/pypi/v/caliscope?color=blue)](https://pypi.org/project/caliscope/)
[![GitHub last commit](https://img.shields.io/github/last-commit/mprib/caliscope.svg)](https://github.com/mprib/caliscope/commits)
[![pytest](https://github.com/mprib/caliscope/actions/workflows/pytest.yml/badge.svg)](https://github.com/mprib/caliscope/actions/workflows/pytest.yml)
</div>

Caliscope is an open-source Python toolkit that calibrates multicamera rigs for 3D motion capture.
From calibration videos, it estimates each camera's focal length and lens distortion, and its position and orientation relative to the other cameras.
Run it from the desktop app, which shows each stage as it goes, or from the scripting API.

## Calibration features

| Feature | What you can do |
|---------|-----------------|
| Partial camera overlap | Calibrate without every camera seeing the target at once. Each camera must link to the rest through shared views of the target. |
| Double-sided targets | Use both faces of a calibration target for cameras facing each other. See the [target guide](https://mprib.github.io/caliscope/calibration_targets/) for how to build and record one. |
| Measured marker distances | Add measured distances between ArUco markers for the solution to respect. |
| Calibration review | Inspect reprojection error, drop outliers, and solve again. |
| Level from video | Estimate which way is up from each camera's video, and rotate the calibration to match. Scripting API only. |
| Calibration export | Save the calibration as TOML, with an aniposelib-compatible copy for [Pose2Sim](https://github.com/perfanalytics/pose2sim) and [anipose](https://anipose.readthedocs.io/). |

In the desktop app, calibrate lenses with a ChArUco board or chessboard, and camera positions with a ChArUco board or ArUco markers.
The scripting API also accepts a chessboard for camera positions.
The measured dimensions of your calibration target set the scale of the result.

An [experimental markerless workflow](https://mprib.github.io/caliscope/markerless_calibration/) calibrates camera positions from a person moving through the scene instead of a target.
It runs from Python after lens calibration, takes scale from one measured distance between two cameras, and levels the rig from the video.

## Install

```bash
uv pip install caliscope             # calibration library and scripting API
uv pip install "caliscope[tracking]" # adds pose tracking and vertical estimation to the API
uv pip install "caliscope[gui]"      # adds the desktop app and 3D viewer, and includes tracking
caliscope                            # launch the app
```

See the [installation guide](https://mprib.github.io/caliscope/installation/) for full setup.
Try the [sample project](https://mprib.github.io/caliscope/sample_project/) with downloadable data.

## Scripting

The scripting API runs the same calibration code as the app, and adds steps the app does not have.
For example, after calibrating, you can level the rig from its own video:

```python
from caliscope.api import CaptureVolume, estimate_vertical

volume = CaptureVolume.load("capture_volume")
videos = {0: "cam_0.mp4", 1: "cam_1.mp4", 2: "cam_2.mp4"}

vertical = estimate_vertical(videos, volume.camera_array)
volume = volume.oriented(up=vertical.up_per_cam)  # +Z now points up
volume.save("capture_volume_level")
```

The vertical estimate comes from a [GeoCalib](https://github.com/cvg/GeoCalib) model converted to ONNX.
It reads up from ordinary images, so expect it to land within a few degrees of true vertical.
It needs the `[tracking]` extra and downloads its model on first use.
The API can also set scale from a tape-measured distance between two cameras, place the floor at zero, and print calibration reports in the terminal.
The [scripting guide](https://mprib.github.io/caliscope/scripting/) covers the full workflow.

## Demo

Recorded with v0.9.0.
The current interface differs.

https://github.com/user-attachments/assets/b8bb78de-866e-4ba2-b5c7-674e3a33dd9e

## Reconstruction

The desktop app can also track landmarks with ONNX pose models, triangulate them in 3D, and export CSV and TRC files.
See [Tracking & Triangulation](https://mprib.github.io/caliscope/reconstruction/) for the workflow.

## Documentation and help

[Full documentation](https://mprib.github.io/caliscope/).
Questions: [Discussions](https://github.com/mprib/caliscope/discussions).
Bugs: [Issues](https://github.com/mprib/caliscope/issues).

## Acknowledgments

Inspired by [anipose](https://github.com/lambdaloop/anipose), created by Lili Karashchuk, PhD.

## License

[BSD 2-Clause](https://opensource.org/license/bsd-2-clause/)
