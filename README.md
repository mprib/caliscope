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

Caliscope is a permissively licensed multicamera calibration tool for markerless motion capture workflows. It allows visual assessment and detailed quality metrics at each stage of the calibration workflow to allow high quality output. The approach to initializing parameters for bundle adjustment ([see docs](https://mprib.github.io/caliscope/extrinsic_calibration/)) allows rapid and reliable calibration.

## Demo

https://github.com/user-attachments/assets/037c6237-0955-41e2-979e-a4247f7677e6

## Quick Start

Installation instructions are in the [docs](https://mprib.github.io/caliscope/installation/).

For a walkthrough with test data after installing, see the [sample project](https://mprib.github.io/caliscope/sample_project/).

---

## Features

#### Calibration

- ChArUco, ArUco, and chessboard calibration targets
- Automated intrinsic calibration from video with distortion model visualization
- Pairwise extrinsic initialization for reliable bundle adjustment across 3+ cameras
- Mirror board support for camera arrangements where no single board position is visible to all cameras
- 3D visualizer for inspecting camera positions and setting the world origin
- Reprojection error display and outlier filtering after optimization
- Exports `camera_array.toml` (native) and `camera_array_aniposelib.toml` for use with [aniposelib](https://github.com/lambdaloop/aniposelib)-compatible tools

#### Tracking and Reconstruction

- Built-in MediaPipe trackers (Hands, Pose, Holistic)
- ONNX model support for custom pose estimators exported from SLEAP, DeepLabCut, RTMPose, and other frameworks
- Output in CSV and TRC (OpenSim) formats

## Community & Support

To report a bug or request a feature, please [open an issue](https://github.com/mprib/caliscope/issues). For questions, post in [Discussions](https://github.com/mprib/caliscope/discussions). This is an open-source project supported by volunteer effort.

## Acknowledgments

This project was inspired by [FreeMoCap](https://github.com/freemocap/freemocap) (FMC), which is spearheaded by [Jon Matthis, PhD](https://jonmatthis.com/) of the HuMoN Research Lab.
The FMC calibration and triangulation system is built upon [Anipose](https://github.com/lambdaloop/anipose), created by Lili Karushchek, PhD.
Caliscope was originally envisioned as an alternative calibration tool to Anipose that would allow more granular estimation and visual feedback.
Several lines of the original Anipose triangulation code are used in this code base, though it was otherwise written from the ground up.

## License

Caliscope is licensed under the permissive [BSD 2-Clause license](https://opensource.org/license/bsd-2-clause/).
The triangulation function was adapted from the [Anipose](https://github.com/lambdaloop/anipose) code base which is also licensed under the BSD-2 Clause.
