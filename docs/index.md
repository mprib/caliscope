# Welcome

Caliscope is a GUI-based, permissively licensed multicamera calibration package. It determines camera intrinsic and extrinsic properties from synchronized video, enabling 3D triangulation of landmark positions for motion capture research.

## What It Does

Given synchronized video from two or more cameras and a calibration target, Caliscope will:

1. **Calibrate each camera's intrinsic properties** (focal length, optical center, lens distortion)
2. **Determine extrinsic properties** (rotation and translation of every camera relative to a common world frame) using pairwise stereo bootstrapping and bundle adjustment
3. **Track 2D landmarks** using built-in MediaPipe trackers or custom ONNX pose estimation models
4. **Triangulate 3D positions** from the 2D observations across cameras
5. **Export results** in `.csv` and `.trc` (OpenSim) formats

## Calibration Targets

Caliscope supports three calibration target types:

- **ChArUco board** — best general-purpose target; works for both intrinsic and extrinsic calibration
- **Chessboard** — simpler alternative for intrinsic calibration
- **ArUco marker** — single printed marker for extrinsic calibration of large capture volumes

See [Calibration Targets](calibration_targets.md) for details on each option.

## Getting Started

The [Installation](installation.md) guide will walk you through setting up the package. [Project Setup](project_setup.md) explains the workspace directory structure and file naming conventions. The calibration and reconstruction guides in the sidebar will take you through each step of the workflow.

A [sample project](sample_project.md) with downloadable data demonstrates the full pipeline.

## Feedback

If you encounter a bug or have a feature request, please [open an issue](https://github.com/mprib/caliscope/issues). For general questions, post in [Discussions](https://github.com/mprib/caliscope/discussions).
