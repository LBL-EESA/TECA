---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Adrian M. Price-Whelan^[Custom footnotes for e.g. denoting who the corresponding author is can be included like this.]
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    affiliation: 2
  - name: Author with no affiliation
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
TECA is a parallel infrastructure for climate analytics designed specifically
for use on DOE HPC supercomputing platforms such as the Cray systems run by
NERSC, ANL, and ORNL.

It's been proven to run at massive scales and is regularly used at
concurrencies exceeding thousands of compute nodes and hundreds of thousands of
CPU cores.  TECA can make use of accelerators such as GPU's through
technologies such as CUDA, OpenCL, OpenMP, and SYCL where available.

# Statement of Need
The analysis of climate extremes requires processing massive amounts of high
spatio-temporal resolution data. These data sets can contain Petabytes and often
span centuries worth of simulated time. Detecting extreme events in such data
is akin to finding a needle in a hay stack. Highly optimized,
efficient, and scalable methods are required to detect and analyze
climatalogical extremes in this data.

# Design and Implementation
TECA contains a collection of highly optimized I/O , computer vision, machine
learning, and general purpose numerical analysis algorithms that can be easily
connected together to construct new climate analysis applications.  TECA is
designed around a pipeline concept where pipeline stages implement specific
functions such as execution control, I/O, data transformation, feature
detection, and so on.

The pipeline abstraction enables the separation of analysis, I/O, and execution
control concerns. This separation of concerns is a key to the system's
flexibility, reusibility and to putting scalable, high performance computing
methods into the hands of climate scientists, who are not always HPC experts.
For instance the reuse of execution control and I/O components enables a
climate to focus on implementing the analysis particular to their specific use
cases without a deep knowledge of HPC, parallel programming, or parallel I/O
techniques.

The core I/O capabilities are leverage MPI independent and collective I/O
either directly or indirectly through MPI, HDF5, and NetCDF.  The core
execution control components of TECA are written in C++ using techgnologies
such as MPI, C++ threads, OpenMP, and CUDA.  Support currently exists for a
diverse set of execution patterns including map reduce over time; map reduce
over space and time; and single program multiple data (SPMD) distributed data
parallel patterns.

<!--
TECA provides a framework for parallel execution where the units of work, the
data to be processed, are presented to the system as an index set. Indices are
mapped to the available hardware such as CPUs and/or GPUs according to rules of
the specific execution pattern in use.  TECA provides execution engines
implementing map reduce and single program multiple data (SPMD) distributed
data execution patterns. 

The creative use of index sets enables parallelization over diverse types of
data. For instance in one use case the indices of an index set might represent
a set of files on disk, in another use case indices might represent spatial
tiles, in another use case an index set might represent the time steps of a
climate simulation, in another case an index set might represent detected
cyclone tracks.

Reductions can be implemented as transformations between two index sets.  Temporal
reductions, which transform the time axis of the data, can be easily
parallelized over the output time axis.  For example, when computing a daily
average from 6 hourly input the input index space has an id for each of the 6
hourly snap shots, while the output index set has an id for each daily average.
The daily average reduction operator effects this transformation.
-->

# Extensibility
TECA is modular and extensible in C++ and Python. Parallelism is built into
modular reusable components and in many cases, new climate analyitcs
applications can leverage existing framework components. When the need for a
new capability, such as a new I/O, execution control, or data processing or
transformation, arises it casn be implemented using C++ or Python.

TECA's pipeline implements a state machine with 3 states:
1. The report state where starting from the top down each component reports on
   the universe of data it could produce.
2. The request state where starting from the bottom up each component requests
   the data it needs to acheive its aims
3. The execute state where starting from the top down each component operates
   on the incoming  data or generates new data 

The procudure for adding a new pipeline component involves providing one or
more functions implementing one or more of the pipeline states.
In the case of both languages, functional and polymorphic approaches are supported.
Coupling to Fortran libraries is also possible as demonstrated by our use of
teh GFDL TC detector [@gfdl_tc].

# Deployment and Application at Scale
TECA includes a number of climatalogical feature detectors such as
detectors for atmospheric rivers (ARs) [@ca_ar, @teca_bard];
detectors for tropical cyclones(TCs) [@gdfl_tc];. Diagnostics such as
the Enso Longitudinal Index[@eli; @teca_eli] have also been implemented.

A number of post fetaure detection analyses are available, such as computing
cyclone sizes using radial wind profiles [@tc_wind_rad] and .
TECA has been used to assess the impacts of a changing climate on extreme events
such as AR's, TC's and ETC's [@ca_ar; @tc_changes; @peta_scale_teca].

TECA provides collection of parallel command line applications that can be
deployed and run on HPC centers. TECA's command line applications are used
to harness super computing systems and has been shown to scale up to full
system concurrencies [@peta_scale_teca].  

TECA's feature detection algorithms and command line applications have been
instrumental in in developing and evaluating a number of machine learning based
detectors [@teca_ml_1; @teca_ml_2; @teca_topo; @ankur_agu].  New statistical
and machine learning detectors have been developed [@teca_bard; @ankur_agu].


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

