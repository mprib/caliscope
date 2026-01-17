# Roadmap

Caliscope is a multicamera calibration system for 3D motion capture. It targets researchers seeking low-cost alternatives to commercial systems like Vicon.

## Current Priority: Calibration Flexibility

The biggest friction point today is the dependency on Charuco boards. For intrinsic calibration, Charuco is overkill — a standard checkerboard suffices and avoids mounting a printed board on a rigid flat surface. For extrinsic calibration, printing a Charuco board large enough to calibrate a big capture volume is impractical.

The plan: checkerboard for intrinsics, ArUco markers for extrinsics.

### Phase 1: Checkerboard Intrinsic Calibration

**Goal**: Support standard checkerboards for intrinsic calibration.

**Why**: Checkerboards are cheap, available in large sizes, and lay flat. Single-camera calibration doesn't require unambiguous rotation determination.

### Phase 2: ArUco-based Extrinsic Calibration

**Goal**: Enable multi-camera extrinsic calibration using ArUco markers instead of Charuco boards.

**Why**: A large ArUco marker on poster paper is visible from distance, cheap to produce, and printable front/back for visibility from any angle. Multiple markers can be scattered around a lab for static calibration. Rigid flatness matters less for extrinsic calibration.

**Workflow**:

- Scatter ArUco markers around the capture volume
- Navigate to frames showing marker visibility across cameras
- Select one marker as world origin
- Adjust orientation (e.g., marker on bookend defines "up" and floor plane)
- Input known edge lengths to establish scale

This mirrors Vicon workflows with $0.10 printed markers.

## Future

### Tracker Integrations

- **SLEAP / DeepLabCut**: Animal behavior research
- **MMPose**: Modern human pose estimation

### Hardware Integration

Frame capture is a separate project. Eventual integration with synchronized camera platforms (webcam clusters, Raspberry Pi arrays) would complete the end-to-end workflow.

### Markerless Biomechanics

Long-term goal: high-fidelity markerless motion capture for biomechanics research. This requires accuracy sufficient to avoid inferring false forces from position noise, plus reliable joint center estimation for inverse dynamics.
