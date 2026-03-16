# Caliscope

Caliscope is a permissively licensed multicamera calibration tool.
It produces the intrinsic and extrinsic camera parameters that downstream tools need for 3D reconstruction.
The objective of this project is to **Do One Thing Well**, and that one thing is estimating camera parameters.

If you sense that there *must* be a good way to calibrate your setup but existing tools don't quite cut it, please [open a discussion](https://github.com/mprib/caliscope/discussions).
I (mprib) am interested in exploring whether Caliscope can be upgraded to accommodate your use case.

## The calibration problem

Multicamera 3D reconstruction requires knowing each camera's optical properties (intrinsic calibration) and its position and orientation in space (extrinsic calibration).
Getting these parameters right is the foundation of accurate triangulation.
Getting them wrong produces errors that propagate silently through every downstream analysis.
Calibrating more than two cameras requires bundle adjustment, a nonlinear optimization that simultaneously refines all camera positions and 3D point estimates.
Bundle adjustment is powerful but sensitive to its starting point: a poor initial estimate can cause it to converge to a poor solution or fail entirely.
When the initial estimate is good, bundle adjustment converges quickly and reliably.
Caliscope is designed to produce that good initial estimate and rapidly solve for a quality calibration. Some important strategies that facilitate this:

- **Pairwise PnP initialization:**
Caliscope builds an initial estimate of camera positions from pairwise relationships using PnP (Perspective-n-Point).
For each pair of cameras that both see the calibration target in the same frame, it estimates their relative position.
It then chains these pairwise estimates transitively: if the relationship between cameras A and B is known, and B to C is known, A to C can be inferred.
This produces a reliable starting point, even when no single position of the target is visible to all cameras at once.

- **Flexible targets:**
The PnP approach allows a single ArUco marker printed on a standard sheet of paper to serve as the extrinsic calibration target.
Large-format prints are visible from greater distances, making it practical to calibrate wide capture volumes.

- **Mirror boards:**
For setups where cameras face inward from all directions (common in animal behavior rigs and dense multicamera arrays), finding board positions visible to every camera can be difficult.
Caliscope supports a charuco board printed on both sides of a rigid surface (mirror image on back), so cameras viewing opposite sides identify the same physical point from either direction.
**This allows cameras to be linked via PnP even if they never share a common view of the board.**

- **Visual feedback:**
During intrinsic calibration, you can inspect the fitted distortion model to catch problems early.
During extrinsic calibration, you can see a 3D representation of the cameras along with the calibration points moving through space.
World scale accuracy is visualized across frames based on the distances between target calibration points at each frame.


## Integration with other tools

Caliscope's primary output is a calibrated camera array: intrinsic parameters (focal length, distortion) and extrinsic parameters (position and orientation) for every camera in your system.
Alongside the native `camera_array.toml`, Caliscope automatically exports `camera_array_aniposelib.toml` in the format used by [aniposelib](https://github.com/lambdaloop/aniposelib).
Tools that consume aniposelib calibrations can use this file directly.

## Tracking and triangulation

Caliscope includes a basic reconstruction pipeline that tracks 2D landmarks and triangulates them into 3D trajectories.
You can load custom ONNX pose estimation models exported from SLEAP, DeepLabCut, RTMPose, or other frameworks.
Output is available in CSV and TRC (OpenSim) formats.
For more complete reconstruction workflows, tools like [anipose](https://anipose.readthedocs.io/) and [Pose2Sim](https://github.com/perfanalytics/pose2sim) will serve you better.
Caliscope's aniposelib-compatible export (see above) makes it straightforward to use Caliscope for calibration and hand off to these tools for downstream processing.

## Getting started

The [Installation](installation.md) guide covers setup.
[Project Setup](project_setup.md) explains the workspace directory structure and file naming conventions.
A [sample project](sample_project.md) with downloadable data demonstrates the full pipeline.

### Scripting workflow

The standard install (`pip install caliscope`) includes intrinsic and extrinsic camera calibration as importable Python functions. Install with `pip install caliscope[gui]` to add the desktop interface and 3D visualization.

The [Scripting API](scripting.md) page walks through the full calibration pipeline from a Python script.

If you encounter a bug or have a feature request, please [open an issue](https://github.com/mprib/caliscope/issues).
For questions, post in [Discussions](https://github.com/mprib/caliscope/discussions).
