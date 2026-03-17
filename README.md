<div align="center">

# Caliscope

*Multicamera Calibration for Motion Capture Workflows*

[![PyPI - Downloads](https://img.shields.io/pypi/dm/caliscope?color=blue)](https://pypi.org/project/caliscope/)
[![PyPI - License](https://img.shields.io/pypi/l/caliscope?color=blue)](https://opensource.org/license/bsd-2-clause/)
[![PyPI - Version](https://img.shields.io/pypi/v/caliscope?color=blue)](https://pypi.org/project/caliscope/)
[![GitHub last commit](https://img.shields.io/github/last-commit/mprib/caliscope.svg)](https://github.com/mprib/caliscope/commits)
[![GitHub stars](https://img.shields.io/github/stars/mprib/caliscope.svg?style=social&label=Star)](https://github.com/mprib/caliscope/stargazers)
![pytest](https://github.com/mprib/caliscope/actions/workflows/pytest.yml/badge.svg)
</div>

Caliscope is a permissively licensed multicamera calibration tool.
It estimates camera intrinsic and extrinsic parameters from synchronized video so that downstream tools can triangulate 3D positions accurately.
Bundle adjustment across 3+ cameras needs a good starting estimate to converge reliably. Caliscope produces that estimate and solves for a quality calibration.

## Demo

https://github.com/user-attachments/assets/b8bb78de-866e-4ba2-b5c7-674e3a33dd9e

## Install

We recommend [uv](https://docs.astral.sh/uv/) for installation.
Full instructions (including uv setup and virtual environments) are in the [docs](https://mprib.github.io/caliscope/installation/).

```bash
# Calibration library
uv pip install caliscope

# Desktop app with 3D visualization
uv pip install caliscope[gui]
```

The base install includes the full calibration pipeline as importable Python functions.
Add `[gui]` for the interactive desktop application.

```bash
# Launch the app (with the virtual environment activated)
caliscope
```

### Development setup

```bash
git clone https://github.com/mprib/caliscope.git
cd caliscope
uv sync --group dev --extra gui
```

## Features

| Feature | What it does |
|---------|-------------|
| Pairwise PnP initialization | Estimates camera positions from stereo pairs chained transitively, so bundle adjustment starts from a reliable point |
| Flexible calibration targets | ChArUco, ArUco, and chessboard targets. A single ArUco marker on a sheet of paper can calibrate a wide volume |
| Mirror board support | A charuco board printed on both sides of a rigid surface links cameras that never share a common view |
| Visual feedback | Inspect distortion models, 3D camera positions, reprojection errors, and world scale accuracy at each stage |
| Outlier filtering | Filter calibration points by reprojection error and re-solve |
| Aniposelib export | Generates `camera_array_aniposelib.toml` for use with [aniposelib](https://github.com/lambdaloop/aniposelib)-compatible tools |

## Scripting API

The base install exposes intrinsic and extrinsic calibration as Python functions with Rich progress bars.
See the [Scripting API docs](https://mprib.github.io/caliscope/scripting/) for a walkthrough.

## Tracking and Reconstruction

Caliscope includes a basic reconstruction pipeline for verifying calibration quality.
You can load ONNX pose estimation models (RTMPose, SLEAP, DeepLabCut, or custom) and export 3D trajectories as CSV or TRC (OpenSim).

For full reconstruction workflows, [anipose](https://anipose.readthedocs.io/) and [Pose2Sim](https://github.com/perfanalytics/pose2sim) are better suited.
Caliscope exports an aniposelib-compatible calibration file, making it simple to calibrate here and hand off to those tools.

## Getting Started

For a walkthrough with sample data, see the [sample project](https://mprib.github.io/caliscope/sample_project/).

## Community & Support

To report a bug or request a feature, [open an issue](https://github.com/mprib/caliscope/issues).
For questions, post in [Discussions](https://github.com/mprib/caliscope/discussions).

## Acknowledgments

Caliscope was inspired by [anipose](https://github.com/lambdaloop/anipose), created by Lili Karashchuk, PhD, which demonstrated the value of accessible multicamera calibration for the research community.

## License

Caliscope is licensed under the [BSD 2-Clause license](https://opensource.org/license/bsd-2-clause/).
