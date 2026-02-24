# Caliscope

Caliscope is an open-source tool for multicamera calibration and 3D triangulation. Any workflow that depends on knowing where your cameras are in space can benefit from it, whether you are tracking animal behavior with [SLEAP](https://sleap.ai/) or [DeepLabCut](https://www.mackenziemathislab.org/deeplabcut), running biomechanical analysis with [Pose2Sim](https://github.com/perfanalytics/pose2sim), or building something else entirely. If you need help fitting Caliscope into your pipeline, please [open a discussion](https://github.com/mprib/caliscope/discussions).

## The calibration problem

Multicamera 3D reconstruction requires knowing each camera's optical properties (intrinsic calibration) and its position and orientation in space (extrinsic calibration). Getting these parameters right is the foundation of accurate triangulation. Getting them wrong produces errors that propagate silently through every downstream analysis.

Extrinsic calibration is the harder of the two. Most approaches require the calibration target to be visible in all cameras simultaneously, which limits how cameras can be arranged. Caliscope takes a different approach: for each pair of cameras that both see the calibration target in the same frame, it estimates their relative position and orientation. It then chains these pairwise estimates together to build a complete camera array, even when no single position of the target is visible to all cameras at once. This initial estimate gives bundle adjustment a good starting point, so it converges reliably.

For surround-view setups where cameras face opposite directions, Caliscope supports mirror boards: a ChArUco pattern printed on both sides of a rigid surface. Cameras viewing opposite sides of the board can calibrate against each other.

The GUI provides feedback throughout the process. During intrinsic calibration, you can inspect the fitted distortion model to catch problems early. During extrinsic calibration, you can see reprojection errors after bundle adjustment, filter outliers, and verify scale accuracy against the known geometry of your calibration target.

## Tracking and triangulation

Once cameras are calibrated, Caliscope can track 2D landmarks and triangulate them into 3D trajectories. You can load custom ONNX pose estimation models exported from SLEAP, DeepLabCut, RTMPose, or other frameworks for tracking specific to your subjects. Built-in MediaPipe trackers are included for convenience. Output is available in CSV and TRC (OpenSim) formats.

These features are downstream of calibration. The core contribution is the calibration system itself.

## Acknowledgments

Caliscope was inspired by [anipose](https://anipose.readthedocs.io/), which demonstrated the value of accessible multicamera calibration for the research community. Caliscope focuses on the calibration step specifically, adding GUI feedback and flexible camera arrangements through pairwise estimation.

## Getting started

The [Installation](installation.md) guide covers setup. [Project Setup](project_setup.md) explains the workspace directory structure and file naming conventions. A [sample project](sample_project.md) with downloadable data demonstrates the full pipeline.

If you encounter a bug or have a feature request, please [open an issue](https://github.com/mprib/caliscope/issues). For questions, post in [Discussions](https://github.com/mprib/caliscope/discussions).
