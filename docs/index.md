# Caliscope

Caliscope calibrates multicamera systems for 3D motion capture.
It estimates intrinsic and extrinsic parameters for every camera in your rig, so downstream tools can triangulate accurately.
BSD-2-Clause licensed.

## What it solves

Triangulating a point from several cameras means knowing each camera's optical properties and where it sits in space.
Measuring that by hand is impractical, so it has to be recovered from video of the rig itself.

The work runs in three stages.
Intrinsic calibration recovers focal length and lens distortion, one camera at a time.
Extrinsic calibration recovers where the cameras sit relative to each other, by bundle adjustment over the observations they share.
Anchoring then moves the solved rig into a frame you can use: vertical pointing up, distances in meters, floor at zero.

Everything is available two ways.
The desktop app gives visual feedback at each stage, and the [Scripting API](scripting.md) runs the same pipeline from Python.

## Calibrating with a target

Print a ChArUco board, an ArUco marker, or a chessboard, and record it moving through the volume.
The target never has to be visible to every camera at once, because Caliscope chains pairwise camera relationships transitively.
A single marker on a sheet of paper can calibrate a wide volume.
For surround rigs where cameras face inward from all sides, a board printed on both faces of a rigid surface links cameras that share no view.

## Calibrating without one

Cameras watching a moving person can be solved from the body alone, with no target.
Pose keypoints stand in for board corners, and the scene supplies the coordinate frame: an estimated vertical, one known distance for scale, and the floor.
This path needs intrinsics solved first, and it runs from the [Scripting API](scripting.md).

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
