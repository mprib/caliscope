# Sample Project

The video below walks through the full calibration and reconstruction workflow. To try this on your own hardware, clone the sample dataset from [caliscope-sample-data](https://github.com/mprib/caliscope-sample-data):

```bash
git clone https://github.com/mprib/caliscope-sample-data.git
```

This sample project uses a 3-camera setup with ArUco marker extrinsic calibration and software synchronization. It contains raw input data only: intrinsic calibration videos, extrinsic calibration videos with timestamps, and one walking recording. You configure calibration targets and run the pipeline yourself following the documentation.

This project illustrates the workflow with a minimal setup. Several improvements would increase the quality of the final results:

- more cameras
- higher resolution and frame rate
- better lighting
- larger calibration board
- hardware synchronization rather than software frame alignment

<iframe width="560" height="315" src="https://www.youtube.com/embed/voE3IKYtuIQ?si=U-ivFqX0trbjG5QA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
