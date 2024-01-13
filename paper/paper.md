---
title: 'Pyxy3D: GUI based multicamera calibration and motion tracking'
tags:
  - Python
  - camera calibration
  - motion capture
authors:
  - name: Donald Prible, PT
    orcid: 0000-0001-9243-468X
    affiliation: 1
  - name: Hao-Yuan Hsiao, PhD
    corresponding: true 
    affiliation: 1
affiliations:
 - name: Institution Name, Country
   index: 2
date: 13 August 2017
bibliography: paper.bib

---

# Summary

Pose estimation via three dimensional motion capture is a critical tool employed in multiple research domains such as rehabilitation, sports science, and robotics.  Pyxy3D is a python package for performing this pose estimation task with pre-recorded footage, allowing various camera hardware setups to be integrated. The entire process is managed by the user via a GUI, allowing visual feedback that can ease troubleshooting and quality control. The calibration workflow allows for estimating intrinsic camera properties (i.e. focal length, optical center, and lens distortion), extrinsic camera properies (i.e. the relative translation and rotation of each camera in space) and setting of the world origin to simplify downstream processing. Pre-existing 2D landmark tracking alogorithms are integrated into the workflow, and are combined with the estimated camera parameters to triangulate 3D landmark position. These 3D trajectories can then be exported to the `.trc` file format for use in biomechanical analysis tools such as OpenSim. 

# Statement of need

Commercially available motion capture systems composed of proprietary hardware and software can be prohibitively expensive. A 10-camera Vicon system, for example, can cost approximately $50,000 @cinematographydatabaseIndieViconSystem2021. This presents a hurdle for researchers, particularly those who study clinical populations with movement disorders that make it challenging to travel from home or a medical facility and to the research lab. Transporting data collection tools to where these clinical populations already are could expand the pool of potential research participants, but presents risks and limitations given the cost of the equipment. Rehabilitation researchers would benefit from an open-source tool that decouples the motion capture data acquisition from the motion capture data processing. This enables the development of lower cost hardware configurations that may be more easily deployed to multiple data collection sites, expanding the scale and ease of data collection.

Machine learning tools that automate 2D landmark tracking have enabled a variety of projects pursuing open source motion capture. DeepLabCut @mathisDeepLabCutMarkerlessPose2018 enables rapid training of custom pose estimation models for use in various animal studies, including humans, and comes included with 3D stereotriangulation using 2 cameras. Moving beyond 2 cameras can improve landmark localization and mitigate challenges of landmark occlusion, though it substantially complicates the process of calibration. While OpenCV @bradskyOpenCVLibrary2000 enables straightforward single and stereocamera calibration, calibrating more than two cameras requires the use of bundle adjustment @mayorovLargescaleBundleAdjustment via a library such as ScipPy @virtanenSciPyFundamentalAlgorithms2020. Anipose @karashchukAniposeToolkitRobust2021 integrates with DeepLabCut and automates the bundle adjustment processs for multicamera calibration as well as the corresponding task of triangulating 3D landmark positions with more than 2 views. To facilitate the calibration process, Anipose makes simplifying assumptions about the underlying camera intrinsic parameters, using 1 parameter for the pinhole camera and 1 parameter for the distortion. An iterative bundle adjustment is used to mitigate the impact of outlier data in the calibration.

Pyxy3D carries forward the triangulation code of Anipose, but provides dedicated estimation of camera intrinsics according to the standard camera model employed by OpenCV (4 parameters for the pinhole camera, 5 parameters for distortion). With more accurate and precise intrinsic parameters, the bundle adjustment computes quickly without need for an iterative approach.

FreeMocap @cherianFreeMoCapFreeOpen2024 is another motion capture package, and it uses the original calibration process developed by Anipose. Unlike Anipose, FreeMocap uses Google's Mediapipe @MediapipeHolistic for pose tracking, which avoids the more complicated setup required by DeepLabCut. Pyxy3D also uses this approach for ease of execution and additionally defines the `Tracker` as an abstract base class that is implemented with multiple versions of Mediapipe tracking (i.e. pose, hands, and holistic). This abstract base class is intended to facilitate ease of integration of additional landmark tracking tools in the future, such as DeepLabCut and MMPose @mmposecontributorsOpenMMLabPoseEstimation2020.

# Acknowledgements

We would like to acknowledge Jon Matthis, PhD who is the lead developer of FreeMocap and who provided valuable early-stage guidance related to general code considerations and open-source project management. Additional thanks to Ryan Govostes for coding contributions and feedback related to the intrinsic calibration GUI development.

# References