# Caliscope

Caliscope calibrates multicamera systems for 3D motion capture.
It estimates intrinsic and extrinsic parameters for every camera in your rig, so downstream tools can triangulate accurately.
BSD-2-Clause licensed.

## What it solves

Multicamera 3D reconstruction requires knowing each camera's optical properties and its position in space.
Bundle adjustment finds these, but it needs a good starting point to converge.
Caliscope builds that starting point and refines it.

For each pair of cameras that both see the calibration target in the same frame, Caliscope estimates their relative position using PnP.
It chains these pairwise estimates transitively: if A-B and B-C are known, A-C is inferred.
The target never needs to be visible to all cameras at once.

This approach supports flexible calibration targets.
A single ArUco marker on a sheet of paper can calibrate a wide capture volume.
For surround setups where cameras face inward from all directions, a charuco board printed on both sides of a rigid surface lets cameras on opposite sides link through shared points.

## Output

Caliscope's output is a calibrated camera array saved as `camera_array.toml`.
It also exports `camera_array_aniposelib.toml` for tools like [Pose2Sim](https://github.com/perfanalytics/pose2sim) and [anipose](https://anipose.readthedocs.io/).

A basic reconstruction pipeline is included for verifying calibration quality and quick landmark export.
For production 3D reconstruction, use Pose2Sim or anipose.

## Getting started

[Installation](installation.md) covers setup.
[Project Setup](project_setup.md) explains the workspace structure.
A [sample project](sample_project.md) with downloadable data demonstrates the full pipeline.
The [Scripting API](scripting.md) covers the calibration pipeline from Python.

Questions: [Discussions](https://github.com/mprib/caliscope/discussions).
Bugs: [Issues](https://github.com/mprib/caliscope/issues).
