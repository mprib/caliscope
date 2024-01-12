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

Pose estimation via three dimensional motion capture is a critical tool employed in multiple research domains such as rehabilitation, sports science, and robotics.  `Pyxy3D` is a python package for performing this pose estimation task with pre-recorded footage, allowing various camera hardware setups to be integrated. The entire process is managed by the user via a GUI, allowing visual feedback that can ease troubleshooting and quality control. The complete workflow allows for estimating intrinsic camera properties (i.e. focal length, optical center, and lens distortion), extrinsic camera properies (i.e. the relative translation and rotation of each camera in space) and setting of the world origin frame of reference to streamline downstream processing. Frames from each camera source are automatically bundled into synchronized batches based on the time of capture. 2D landmark tracking is then combined with the estimated camera parameters to estimate 3D landmark position. These 3D trajectories can then be exported to the `.trc` file format for use in biomechanical analysis tools such as OpenSim. `Pyxy3D` defines the tracker requirements as an abstract base class allowing for ease of integrating additional tracking tools in the future, and comes included with an implentation using Google's Mediapipe that allows single-subject human pose estimation.


Another hurdle is related to software. The intrinsic properties of each camera as well as their spatial relationship to each other must be inferred from video recorded with a calibration board or wand. For each motion capture trial, tracked landmarks must be identified and properly labelled in every 2D image. All labelled 2D landmarks must then be combined with the camera properties and their estimated positions to estimate 3D landmark position. While open source software tools are available for each step of this problem, integrating these stages of processing can present an overwhelming burden for researchers who would like to avoid the expense of commercially available systems.

# Statement of need

Commercially available motion capture systems composed of proprietary hardware and software can be prohibitively expensive. A 10-camera Vicon system can cost approximately $50,000 @cinematographydatabaseIndieViconSystem2021. This presents a hurdle for researchers, particularly those who study clinical populations with movement disorders that make travel from the home or medical facility and to the research lab a challenge. Transporting data collection tools to where these clinical populations already are could expand the pool of potential research participants, but presents risks given the cost of the equipment. Rehabilitation researchers would benefit from an open-source tool that decouples the mocap data acquisition from the mocap data processing. This would enable development of lower cost hardware configurations that may be more easily deployed to a variety of settings.

The challenge of integrating various open-source tools to perform the required steps of calibration, tracking, frame synchronization and triangulation limits the ability to explore alternative lower cost hardware configurations. `Pyxy3D` automates these tasks 




Commercially available motion capture systems, such as Vicon, Qualisys, or OptiTrack are expensive. , presenting a hurdle to research. While synchronized frames can be acquired from multiple cameras using a shared external trigger, processsing those frames using various available open-source tools to extract 3D landmark estimates is a substantial challenge. Pyxy3D is inteded to streamline this processing task so that researchers can acquire motion capture data from less expensive hardware. 

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