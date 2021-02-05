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
  - HDF5
  - NetCDF
  - parallel I/O
authors:
  - name: Burlen Loring^[Custom footnotes for e.g. denoting who the corresponding author is can be included like this.]
    orcid: 0000-0002-4678-8142
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
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
Analysis of climate extremes requires processing massive amounts of high
spatio-temporal resolution data. Data sets can contain Peta bytes and often
span centuries worth of simulated time. Detecting extreme events is akin to
finding a needle in a hay stack. Highly optimized, efficient, and scalable
methods are required to detect and analyze climatalogical extremes in this data.

TECA is an optimized, efficient, scalable infrastructure for climate analytics.
It's been proven to run at amssive scales and is regularly used at
concurrencies exceeding thousands of compute nodes and 100k CPU cores.

TECA is designed specifically for climate analytics on doe hpc supercomputing platforms.

TECA is a number of things...

1. A framework for parallel execution , with support for diverse execution patterns eg map reduce over time, map reduce over space and time, and spmd - distributed data parallel paterns
2. A collection of highly optimized I/o , computer vision, machine learning, and general purpose numerical analysis algorithms that can be easily connected together to construct new climate analysis applications.
3. A collection of parallel command line applications that can be deployed and run on hpc centers. This includes Tc and ar detectors, and post event detection analysis applications
4. TECA is modular and extensible in c++ and python. Parallelism is built into modular reusable components and in most cases, new applications can leverage these framework components without the need to deal with the details of parallel programming.

What is TECA's future?
In it's current iteration teca makes use of mpi plus threads to take full advantage of cpu based systems. we are working to expand the design to include gpus and accelerators. Technologies that support a hardware portable task based programming model such as kokkos and cycl will be key. We are currently working on prototypes using these technologies.

# Statement of need

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

