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
From calibration videos, it estimates each camera's focal length and lens distortion, as well as its position and orientation relative to the other cameras.
Run it from the desktop app, which visualizes each stage of processing, or from the scripting API for routine workflows.

## Calibration features

| Feature | What you can do |
|---------|-----------------|
| Partial camera overlap | Calibrate without every camera seeing the target at once. Each camera can link to the rest by chaining together links formed from common views. |
| Double-sided targets | Use both faces of a calibration target for cameras facing each other. See the [target guide](https://mprib.github.io/caliscope/calibration_targets/) for how to build and configure one. |
| Measured marker distances | Add measured distances between ArUco markers for the optimizer to fit to. |
| Calibration review | Inspect overlapping calibration board views, reprojection error, and scale accuracy of the calibration target. |
| Align to calibration board origin | Set the world origin and axes from the board's position in a chosen video frame. |
| Align to vertical from video | Estimate which way is up from each camera's video. Scripting API only. |
| Calibration export | Save the calibration as TOML, with an aniposelib-compatible copy for [Pose2Sim](https://github.com/perfanalytics/pose2sim) and [anipose](https://anipose.readthedocs.io/). |

In the desktop app, calibrate lenses with a ChArUco board or chessboard, and camera positions with a ChArUco board or ArUco markers.
The scripting API also accepts a chessboard for camera positions.
The measured dimensions of your calibration target set the scale of the result.

An [experimental markerless workflow](https://mprib.github.io/caliscope/markerless_calibration/) calibrates camera positions from a person moving through the scene instead of a target (lens calibration required in advance).

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

Vertical estimates come from a [GeoCalib](https://github.com/cvg/GeoCalib) model converted to ONNX.
Because they work from images alone, expect them to land within a few degrees of true vertical.
They need the `[tracking]` extra, and the model downloads on first use.
The [scripting guide](https://mprib.github.io/caliscope/scripting/) covers the full workflow, including setting world scale and the floor plane without a board.

## Demo

Recorded with v0.9.0.
The current interface differs.

https://github.com/user-attachments/assets/b8bb78de-866e-4ba2-b5c7-674e3a33dd9e

## Reconstruction

To check calibration quality, the desktop app can also track landmarks with ONNX pose models, triangulate them in 3D, and export CSV and TRC files.
See [Tracking & Triangulation](https://mprib.github.io/caliscope/reconstruction/) for the workflow.
For anything beyond a quality check, use a dedicated tool such as [Pose2Sim](https://github.com/perfanalytics/pose2sim).

## Documentation and help

[Full documentation](https://mprib.github.io/caliscope/).
Questions: [Discussions](https://github.com/mprib/caliscope/discussions).
Bugs: [Issues](https://github.com/mprib/caliscope/issues).

## Acknowledgments

Inspired by [anipose](https://github.com/lambdaloop/anipose), created by Lili Karashchuk, PhD.

## License

[BSD 2-Clause](https://opensource.org/license/bsd-2-clause/)
