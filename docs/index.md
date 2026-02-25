# Caliscope

Caliscope is an open-source multicamera calibration tool. It produces the intrinsic and extrinsic camera parameters that downstream tools need for 3D reconstruction, whether you are tracking animal behavior with [SLEAP](https://sleap.ai/) or [DeepLabCut](https://www.mackenziemathislab.org/deeplabcut), running biomechanical analysis with [Pose2Sim](https://github.com/perfanalytics/pose2sim), or building something else entirely. If you need help fitting Caliscope into your pipeline, please [open a discussion](https://github.com/mprib/caliscope/discussions).

## The calibration problem

Multicamera 3D reconstruction requires knowing each camera's optical properties (intrinsic calibration) and its position and orientation in space (extrinsic calibration). Getting these parameters right is the foundation of accurate triangulation. Getting them wrong produces errors that propagate silently through every downstream analysis.

OpenCV provides functions for stereo calibration between two cameras, but most practical setups benefit from three or more. Additional viewpoints reduce occlusion and improve triangulation accuracy. Calibrating more than two cameras requires bundle adjustment, a nonlinear optimization that simultaneously refines all camera positions and 3D point estimates. Bundle adjustment is powerful but sensitive to its starting point: a poor initial estimate can cause it to converge to a wrong solution or fail entirely.

Caliscope addresses this by building an initial estimate from pairwise relationships. For each pair of cameras that both see the calibration target in the same frame, it estimates their relative position using PnP (Perspective-n-Point) estimation, a general technique that does not depend on OpenCV's stereo calibration. It then chains these pairwise estimates together transitively: if the relationship between cameras A and B is known, and B to C is known, A to C can be inferred. This produces a rough but reliable starting point for bundle adjustment, even when no single position of the target is visible to all cameras at once.

This pairwise approach also opens up flexibility in calibration targets. A single ArUco marker printed on a standard sheet of paper can serve as the extrinsic target. Large-format prints are visible from greater distances, making it practical to calibrate wide capture volumes. For setups where cameras face inward from all directions (common in animal behavior rigs and dense multicamera arrays), finding board positions visible to every camera can be difficult. Caliscope supports mirror boards: a calibration pattern printed on both sides of a rigid surface, so cameras viewing opposite sides identify the same physical point from either direction. This enables calibration of camera arrangements that would be difficult to handle with existing tools.

The GUI provides feedback throughout the process. During intrinsic calibration, you can inspect the fitted distortion model to catch problems early. During extrinsic calibration, you can see reprojection errors after bundle adjustment, filter outliers, and verify scale accuracy against the known geometry of your calibration target.

## Integration with other tools

Caliscope's primary output is a calibrated camera array: intrinsic parameters (focal length, distortion) and extrinsic parameters (position and orientation) for every camera in your system. Alongside the native `camera_array.toml`, Caliscope automatically exports `camera_array_aniposelib.toml` in the format used by [aniposelib](https://github.com/lambdaloop/aniposelib). Tools that consume aniposelib calibrations can use this file directly.

## Tracking and triangulation

Caliscope includes a basic reconstruction pipeline that tracks 2D landmarks and triangulates them into 3D trajectories as a proof of concept. You can load custom ONNX pose estimation models exported from SLEAP, DeepLabCut, RTMPose, or other frameworks. Built-in MediaPipe trackers are included for convenience. Output is available in CSV and TRC (OpenSim) formats. Signal processing (gap filling, smoothing, filtering) and biomechanical modeling are outside the current scope.

## Acknowledgments

Caliscope was inspired by [anipose](https://anipose.readthedocs.io/), which demonstrated the value of accessible multicamera calibration for the research community. Caliscope focuses on providing GUI feedback to identify problems in the calibration process and on using granular parameter initialization to improve the likelihood of calibration success.

## Getting started

The [Installation](installation.md) guide covers setup. [Project Setup](project_setup.md) explains the workspace directory structure and file naming conventions. A [sample project](sample_project.md) with downloadable data demonstrates the full pipeline.

If you encounter a bug or have a feature request, please [open an issue](https://github.com/mprib/caliscope/issues). For questions, post in [Discussions](https://github.com/mprib/caliscope/discussions).
