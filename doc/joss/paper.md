---
title: 'TECA: The Toolkit for Extreme Climate Analysis'
tags:
  - C++
  - Python
  - Fortran
  - climateology
  - extreme event detection and attribution
  - climate science
  - big data
  - machine learning
  - parallel processing
  - high performance computing
  - GPU
  - MPI
  - OpenMP
  - threading
  - CUDA
  - HDF5
  - NetCDF
  - parallel I/O
authors:
  - name: Burlen Loring^[Custom footnotes for e.g. denoting who the corresponding author is can be included like this.]
    orcid: 0000-0002-4678-8142
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Travis O'Brien
    orcid:
    affiliation:
  - name: Prabhat
    orcid:
    affiliation:
  - name: Suren Byna
    orcid:
    affiliation:
  - name: Oliver Ruebel
    orcid:
    affiliation:
  - name: Abdel Elbashandy
    orcid:
    affiliation:
  - name: Jefferey N Johnson
    orcid:
    affiliation:
  - name: Harinarayan Krishnan
    orcid:
    affiliation:
  - name: William Collins
    orcid:
    affiliation:
  - name: Michael Whener
    orcid:
    affiliation:

affiliations:
 - name: Lawrence Berkeley National Lab
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
   index: 3
date: Fri Feb  5 10:24:36 AM PST 2021
bibliography: paper.bib

# Summary
TECA is an optimized, efficient, scalable infrastructure for climate analytics.
It's been proven to run at massive scales and is regularly used at
concurrencies exceeding thousands of compute nodes and 100k CPU cores.
TECA can make use of accelerators such as GPU's through technologies
such as CUDA, OpenCL, OpenMP, and SYCL where available.

# Statement of need
Analysis of climate extremes requires processing massive amounts of high
spatio-temporal resolution data. Data sets can contain Peta bytes and often
span centuries worth of simulated time. Detecting extreme events in these
datasets is akin to finding a needle in a hay stack. Highly optimized,
efficient, and scalable methods are required to detect and analyze
climatalogical extremes in this data.


TECA is designed specifically for climate analytics on DOE HPC supercomputing platforms
such as the Cray systems run by NERSC, ANL, and ORNL.

TECA is a number of things...

1. A framework for parallel execution , with support for diverse execution
   patterns eg map reduce over time, map reduce over space and/or time, and spmd -
   distributed data parallel paterns

2. A collection of highly optimized I/O , computer vision, machine learning,
   and general purpose numerical analysis algorithms that can be easily connected
   together to construct new climate analysis applications.

3. A collection of parallel command line applications that can be deployed and
   run on hpc centers. This includes Tc and ar detectors, and post event detection
   analysis applications

TECA is modular and extensible in C++ and Python. Parallelism is built into
modular reusable components and in most cases, new applications can leverage
these framework components without the need to deal with the details of
parallel programming.

TECA has been used to asses the impacts of climate change
 \cite{peta_scale_teca} 
 \cite{tc_changes}
and in addtition to many image processing, I/O, and general purpose numerical
methods includes topological, statistical, and machine learning based methods
\cite{teca_bard}.
TECA has been used as the basis for applying new diagnostic techniques such as
the ELI \cite{eli, teca_eli} at scale.


@InProceedings{peta_scale_teca,
@Article{ar_topo,
@Article{tc_changes,
@Article{teca_bard,
@Article{ca_ar,
@article{ruebel,
@inproceedings{pre_teca_ar,
@misc{teca_ml_1,
@misc{teca_ml_2,




# Citations

# Acknowledgements

This research was supported by the Director, Office of Science, Office of
Biological and Environmental Research of the US Department of Energy Regional
and Global Climate Modeling Program (RGCM) and used resources of the National
Energy Research Scientific Computing Center (NERSC), also supported by the
Office of Science of the US Department of Energy under contract no.
DE-AC02-05CH11231. The authors thank Christopher J. Paciorek for providing
useful input on the manuscript. The authors would like to express their sincere
gratitude for input from two anonymous reviewers, whose comments greatly
improved the presentation of the methodology and the resulting discussion.

This research has been supported by the Department of Energy (grant no.
ESD13052).

