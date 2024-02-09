---
title: 'Caliscope: GUI based multicamera calibration and motion tracking'
tags:
  - Python
  - camera calibration
  - motion capture
authors:
  - name: Donald Prible
    orcid: 0000-0001-9243-468X
    affiliation: 1
affiliations:
 - name: The University of Texas at Austin, United States
   index: 1
date: 31 January 2024
bibliography: paper.bib

---

# Summary

3D motion capture is an indispensable tool employed in multiple research domains such as rehabilitation, sports science, and robotics.  This technique necessitates a thorough calibration of each camera's intrinsic properties, the alignment of all cameras in a shared spatial context and accurate 2D point tracking. With these calculations, it is possible to triangulate landmark locations and reconstruct movement dynamics. Historically, this process has relied on costly proprietary tools, though the emergence of open-source pose estimation tools and the availability of high-quality consumer-grade cameras have paved the way for innovative approaches.

Caliscope is a Python package designed to automate the process of camera calibration and landmark triangulation using pre-recorded video, enabling a range of camera hardware setups and tracking solutions to be employed. User interaction with the software is facilitated through a graphical user interface (GUI), which provides visual feedback to assist in troubleshooting and quality control. 


# Statement of need

Commercially available motion capture systems composed of proprietary hardware and software can be prohibitively expensive. A 10-camera Vicon system, for example, can cost approximately $50,000 [@cinematographydatabaseIndieViconSystem2021]. This presents a hurdle for researchers, particularly those who study clinical populations with movement disorders that make travel to a research lab challenging. Transporting data collection tools to where these clinical populations already are could expand the pool of potential research participants, but presents risks and limitations given the cost of the equipment. Rehabilitation researchers would benefit from an open-source tool that decouples the motion capture data acquisition from the motion capture data processing. This enables the development of lower cost hardware configurations that may be more easily deployed to multiple data collection sites, expanding the scale and ease of data collection.

The emergence of machine learning tools for automated 2D landmark tracking has facilitated a variety of open-source motion capture projects. 
DeepLabCut [@mathisDeepLabCutMarkerlessPose2018], for example, allows for the rapid training of custom pose estimation models, applicable to a range of animal subjects, including humans. It offers 3D stereotriangulation using two cameras. 

Expanding beyond two cameras enhances landmark localization and mitigates landmark occlusion but adds significant complexity to the calibration process.
While OpenCV [@bradskyOpenCVLibrary2000] enables straightforward single and stereocamera calibration, calibrating more than two cameras requires the use of bundle adjustment [@mayorovLargescaleBundleAdjustment], leveraging optimization tools such as ScipPy [@virtanenSciPyFundamentalAlgorithms2020].

Anipose [@karashchukAniposeToolkitRobust2021], which integrates with DeepLabCut, automates the bundle adjustment processs for multicamera calibration and the triangulation of 3D landmark positions from more than 2 views. 
To facilitate the calibration process, Anipose makes simplifying assumptions about the underlying camera intrinsic parameters, using 1 parameter for the pinhole camera and 1 parameter for the distortion. A process of iterative bundle adjustment is used to remove the impact of outlier results in the calibration.

Caliscope carries forward the triangulation code of Anipose, but provides dedicated estimation of camera intrinsics according to the standard camera model employed by OpenCV[@bradskyOpenCVLibrary2000]. It therefore estimates 4 parameters for the pinhole camera model, and 5 parameters for the lens distortion. This is in contrast to the 2 parameters estimated by Anipose (focal length and a single distortion paramters). With more accurate and precise intrinsic parameters, the bundle adjustment computes quickly without need for an iterative approach.

FreeMocap [@cherianFreeMoCapFreeOpen2024] is another notable open-source motion capture project. It uses the original calibration process developed by Anipose. Unlike Anipose, FreeMocap uses Google's Mediapipe [@MediapipeHolistic] for pose tracking, which avoids the more complicated setup required by DeepLabCut. Caliscope also uses this approach for ease of execution and additionally defines the `Tracker` as an abstract base class that is implemented with multiple versions of Mediapipe tracking (i.e. pose, hands, and holistic). This abstract base class is intended to facilitate ease of integration of additional landmark tracking tools in the future, such as MMPose [@mmposecontributorsOpenMMLabPoseEstimation2020].

# Acknowledgements

The author would like to extend gratitude to Lili Karushcheck, PhD whose project Anipose provided the initial inspiration for Caliscope as well as to Jon Matthis, PhD who is the lead developer of FreeMocap. Dr. Matthis provided valuable early-stage guidance related to general code considerations and open-source project management. Additional thanks are due to Ryan Govostes for coding contributions and feedback related to the intrinsic calibration GUI development.

# References
