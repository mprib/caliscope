# Sample Project

The video below walks through the full calibration and reconstruction workflow. To try this on your own hardware, download the sample dataset [here](https://1drv.ms/f/c/a30b139c66ff49c7/EqIVjIRLQ9hEh7hLE7UysAcBvxa1Oqy8JlM8Cu1gg0mXKw?e=3PAYXa).

This sample project uses a 3-camera setup with ChArUco board calibration and MediaPipe pose tracking. It demonstrates the full pipeline: intrinsic calibration, extrinsic calibration, and 3D reconstruction.

This project illustrates the workflow with a minimal setup. Several improvements would increase the quality of the final results:

- more cameras
- higher resolution and frame rate
- better lighting
- larger calibration board
- hardware synchronization rather than software frame alignment

<iframe width="560" height="315" src="https://www.youtube.com/embed/voE3IKYtuIQ?si=U-ivFqX0trbjG5QA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
