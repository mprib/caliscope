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
```
    Summary: Has a clear description of the high-level functionality and purpose of the software for a diverse, non-specialist audience been provided?

    A statement of need: Does the paper have a section titled ‘Statement of need’ that clearly states what problems the software is designed to solve, who the target audience is, and its relation to other work?

    State of the field: Do the authors describe how this software compares to other commonly-used packages?

    Quality of writing: Is the paper well written (i.e., it does not require editing for structure, language, or writing quality)?

    References: Is the list of references complete, and is everything cited appropriately that should be cited (e.g., papers, datasets, software)? Do references in the text use the proper citation syntax?

```

# Summary

Pose estimation via three dimensional motion capture is a critical tool employed in multiple research domains such as rehabilitation, sports science, and robotics.  `Pyxy3D` is a python package for performing this pose estimation task with pre-recorded footage provided that the time of capture is stored for each frame. This allows various camera hardware setups to be integrated. The entire process is managed by the user via a GUI, allowing visual feedback that can ease troubleshooting and quality control. The calibration workflow allows for estimating intrinsic camera properties (i.e. focal length, optical center, and lens distortion), extrinsic camera properies (i.e. the relative translation and rotation of each camera in space) and setting of the world origin frame of reference to streamline downstream processing. 2D landmark tracking is then combined with the estimated camera parameters to triangulate 3D landmark position. These 3D trajectories can then be exported to the `.trc` file format for use in biomechanical analysis tools such as OpenSim. 

# Statement of need

Commercially available motion capture systems composed of proprietary hardware and software can be prohibitively expensive. A 10-camera Vicon system, for example, can cost approximately $50,000 @cinematographydatabaseIndieViconSystem2021. This presents a hurdle for researchers, particularly those who study clinical populations with movement disorders that make it challenging to travel from home or a medical facility and to the research lab. Transporting data collection tools to where these clinical populations already are could expand the pool of potential research participants, but presents risks and limitations given the cost of the equipment. Rehabilitation researchers would benefit from an open-source tool that decouples the mocap data acquisition from the mocap data processing. This enables development of lower cost hardware configurations that may be more easily deployed to multiple data collection sites, expanding the scale and ease of data collection.

Machine learning tools that automate 2D landmark tracking have enabled a variety of projects pursuing open source motion capture. DeepLabCut @mathisDeepLabCutMarkerlessPose2018 enables rapid training of custom pose estimation models for use in various animal studies, including humans, and comes included with the triangulation from a pair of cameras. Additional cameras can improve landmark localization and mitigate challenges of landmark occlusion, though it substantially complicates the process of camera calibration. While OpenCV @bradskyOpenCVLibrary2000 enables straightforward single and stereo camera calibration, calibrating more than two cameras requires the use of bundle adjustment @mayorovLargescaleBundleAdjustment using a library such as ScipPy @virtanenSciPyFundamentalAlgorithms2020. Anipose @karashchukAniposeToolkitRobust2021 integrates with DeepLabCut and automates the bundle adjustment processs for multicamera calibration as well as the corresponding task of triangulating 3D landmark position with more than 2 views. To facilitate the calibration process, Anipose makes simplifying assumptions about the underlying camera intrinsic parameters and employs an iterative bundle adjustment to mitigate the impact of outlier data.

Pyxy3D carries forward the triangulation code of Anipose, but provides dedicated estimation of camera intrinsics according to the standard camera model employed by OpenCV. With more accurate and precise intrinsics, the bundle adjustment proceeds quickly without need for an iterative approach. Rather than 


`Pyxy3D` defines the tracker requirements as an abstract base class allowing for ease of integrating additional tracking tools in the future, and comes included with an implentation using Google's Mediapipe that allows single-subject human pose estimation.

## State of the field

>`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.


# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References