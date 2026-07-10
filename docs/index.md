# Caliscope

Caliscope calibrates multicamera systems for 3D motion capture.
It figures out where each camera is and how its lens behaves, so downstream tools can triangulate accurately.
BSD-2-Clause licensed.

## What it does

You record video of a calibration target from your cameras, and Caliscope computes the camera parameters.
The target does not need to be visible to all cameras at once.
As long as pairs of cameras share a view of the target at some point, Caliscope can link them together.

Caliscope works with simple printed targets.
A ChArUco board handles most setups.
For large volumes, a single ArUco marker on a sheet of paper is enough.
For surround setups where cameras face inward from all directions, a board printed on both sides links cameras that never share a direct view.

## Output

Caliscope saves calibration as `camera_array.toml` and exports an aniposelib-compatible file for tools like [Pose2Sim](https://github.com/perfanalytics/pose2sim) and [anipose](https://anipose.readthedocs.io/).

A basic reconstruction pipeline is included for verifying calibration quality.
For production 3D reconstruction, use Pose2Sim or anipose.

## Getting started

[Installation](installation.md) covers setup.
[Project Setup](project_setup.md) explains the workspace structure.
A [sample project](sample_project.md) with downloadable data demonstrates the full pipeline.
The [Scripting API](scripting.md) covers calibration from Python.

Questions: [Discussions](https://github.com/mprib/caliscope/discussions).
Bugs: [Issues](https://github.com/mprib/caliscope/issues).
