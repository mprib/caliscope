<div align="center">

# Caliscope

*Multicamera Calibration for Motion Capture*

[![PyPI - Downloads](https://img.shields.io/pypi/dm/caliscope?color=blue)](https://pypi.org/project/caliscope/)
[![PyPI - License](https://img.shields.io/pypi/l/caliscope?color=blue)](https://opensource.org/license/bsd-2-clause/)
[![PyPI - Version](https://img.shields.io/pypi/v/caliscope?color=blue)](https://pypi.org/project/caliscope/)
[![GitHub last commit](https://img.shields.io/github/last-commit/mprib/caliscope.svg)](https://github.com/mprib/caliscope/commits)
![pytest](https://github.com/mprib/caliscope/actions/workflows/pytest.yml/badge.svg)
</div>

3D motion capture requires knowing each camera's intrinsic properties (focal length and lens distortion), as well as where the cameras sit relative to each other in space.
With these parameters, observations from multiple cameras can be triangulated into accurate 3D coordinates.

Caliscope efficiently estimates these parameters with a scriptable Python API as well as a GUI that offers visual feedback during each stage of the calibration.


## Demo

Note that the workflow below is from v0.9.0. There have recently been a number of improvements. I hope to have more recent samples/walkthroughs soon.

https://github.com/user-attachments/assets/b8bb78de-866e-4ba2-b5c7-674e3a33dd9e

## Install

```bash
uv pip install caliscope        # calibration library + scripting API
uv pip install caliscope[gui]   # adds desktop app, 3D visualization, and pose tracking
caliscope                       # launch the app
```

Full instructions at [mprib.github.io/caliscope/installation](https://mprib.github.io/caliscope/installation/).

## Features

| Feature | What it does |
|---------|-------------|
| Pairwise PnP initialization | Chains stereo pair estimates transitively, so the target never needs to be visible to all cameras at once |
| Flexible targets | ChArUco boards, ArUco markers, and chessboards. A single marker on a sheet of paper can calibrate a wide volume |
| Mirror boards | A board printed on both sides links cameras that never share a common view |
| Outlier filtering | Filter by reprojection error and re-solve |
| Aniposelib export | `camera_array_aniposelib.toml` for [Pose2Sim](https://github.com/perfanalytics/pose2sim), [anipose](https://anipose.readthedocs.io/), and other aniposelib-compatible tools |

## Scripting API

The base install exposes calibration as Python functions.
See the [scripting docs](https://mprib.github.io/caliscope/scripting/) for a walkthrough.

## Reconstruction

A basic reconstruction pipeline is included for verifying calibration quality: ONNX pose tracking, triangulation, and CSV/TRC export.
For production reconstruction, use Pose2Sim or anipose with Caliscope's exported calibration.

## Getting started

[Sample project](https://mprib.github.io/caliscope/sample_project/) with downloadable data.
[Full documentation](https://mprib.github.io/caliscope/).

Bugs: [Issues](https://github.com/mprib/caliscope/issues). Questions: [Discussions](https://github.com/mprib/caliscope/discussions).

## Acknowledgments

Inspired by [anipose](https://github.com/lambdaloop/anipose), created by Lili Karashchuk, PhD.

## License

[BSD 2-Clause](https://opensource.org/license/bsd-2-clause/)
