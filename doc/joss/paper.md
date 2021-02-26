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
TECA is an parallel infrastructure for climate analytics designed specifically
for use on DOE HPC supercomputing platforms such as the Cray systems run by
NERSC, ANL, and ORNL.

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

TECA provides a framework for parallel execution , with support for diverse execution
patterns eg map reduce over time, map reduce over space and/or time, and SPMD -
distributed data parallel patterns

TECA contains a collection of highly optimized I/O , computer vision, machine
learning, and general purpose numerical analysis algorithms that can be easily
connected together to construct new climate analysis applications.  TECA is
designed around a pipeline concept where pipeline stages implement specific
functions such as execution control, I/O, data transformation, feature
detection, and so on. Packaging specific functionality into modular connectable
objects enables reuse.

TECA is modular and extensible in C++ and Python. Parallelism is built into
modular reusable components and in most cases, new applications can leverage
these framework components without the need to deal with the details of
parallel programming.

TECA also provides collection of parallel command line applications that can be
deployed and run on HPC centers. This includes tropical cyclone(TC) and
atmospheric river(AR) detectors, and post event detection analysis applications

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

