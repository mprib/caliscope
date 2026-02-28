<div align="center">

# Caliscope

*Multicamera Calibration for Research Workflows*

[![PyPI - Downloads](https://img.shields.io/pypi/dm/caliscope?color=blue)](https://pypi.org/project/caliscope/)
[![PyPI - License](https://img.shields.io/pypi/l/caliscope?color=blue)](https://opensource.org/license/bsd-2-clause/)
[![PyPI - Version](https://img.shields.io/pypi/v/caliscope?color=blue)](https://pypi.org/project/caliscope/)
[![GitHub last commit](https://img.shields.io/github/last-commit/mprib/caliscope.svg)](https://github.com/mprib/caliscope/commits)
[![GitHub stars](https://img.shields.io/github/stars/mprib/caliscope.svg?style=social&label=Star)](https://github.com/mprib/caliscope/stargazers)
![pytest](https://github.com/mprib/caliscope/actions/workflows/pytest.yml/badge.svg)
</div>

Caliscope is a permissively licensed multicamera calibration tool.
Bundle adjustment across 3+ cameras requires a good initial estimate of camera positions to converge quickly and reliably.
Caliscope is designed to produce that good initial estimate and rapidly solve for a quality calibration.

## Demo

https://github.com/user-attachments/assets/b8bb78de-866e-4ba2-b5c7-674e3a33dd9e

## Quick Start

Installation instructions are in the [docs](https://mprib.github.io/caliscope/installation/).

For a walkthrough with test data after installing, see the [sample project](https://mprib.github.io/caliscope/sample_project/).

---

## Features

| Feature | What it does |
|---------|-------------|
| Pairwise PnP initialization | Estimates camera positions from stereopairs chained transitively, so bundle adjustment starts from a reliable point |
| Flexible calibration targets | ChArUco, ArUco, and chessboard targets; a single ArUco marker on a sheet of paper can calibrate a wide volume |
| Mirror board support | A charuco board printed on both sides of a rigid surface links cameras that never share a common view |
| Visual feedback | Inspect distortion models, 3D camera positions, reprojection errors, and world scale accuracy at each stage |
| Outlier filtering | Filter calibration points by reprojection error after optimization and re-solve |
| Aniposelib export | Automatically generates `camera_array_aniposelib.toml` for use with [aniposelib](https://github.com/lambdaloop/aniposelib)-compatible tools |

## Tracking and Reconstruction

Caliscope includes a basic reconstruction pipeline for verifying calibration quality.
You can load ONNX pose estimation models (RTMPose, SLEAP, DeepLabCut, or custom) and export 3D trajectories in CSV and TRC (OpenSim) formats.
For more complete reconstruction workflows, tools like [anipose](https://anipose.readthedocs.io/) and [Pose2Sim](https://github.com/perfanalytics/pose2sim) will serve you better.
The aniposelib-compatible export makes it straightforward to use Caliscope for calibration and hand off to these tools for downstream processing.

## Community & Support

To report a bug or request a feature, please [open an issue](https://github.com/mprib/caliscope/issues).
For questions, post in [Discussions](https://github.com/mprib/caliscope/discussions).

## Acknowledgments

Caliscope was inspired by [anipose](https://github.com/lambdaloop/anipose), created by Lili Karashchuk, PhD, which demonstrated the value of accessible multicamera calibration for the research community.
Several lines of the original anipose triangulation code are used in this code base.

## License

Caliscope is licensed under the [BSD 2-Clause license](https://opensource.org/license/bsd-2-clause/).
The triangulation function was adapted from [anipose](https://github.com/lambdaloop/anipose), also licensed under BSD-2-Clause.
