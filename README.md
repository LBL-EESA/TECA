<table style="border: 1px; border-collapse: collapse; border-spacing: 0px;">
<tr><td width="402px">
<img src="https://raw.githubusercontent.com/LBL-EESA/TECA/8ea6a121c8c29cdab31be4226b0564c9ee5a9726/doc/rtd/images/teca_logo_crop2_lg.png" width="400px">
</td></tr>
</table>
<a href="https://travis-ci.com/LBL-EESA/TECA"><img src="https://travis-ci.com/LBL-EESA/TECA.svg?token=zV3LhFtYvjcvo67W2uji&branch=master"></a>
<a href="https://teca.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/teca/badge/?version=latest"></a>
<a href="https://badge.fury.io/py/teca"><img src="https://badge.fury.io/py/teca.svg" alt="PyPI version"></a>
<a href="https://doi.org/10.5281/zenodo.6640287"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6640287.svg" alt="DOI"></a>

## The Toolkit for Extreme Climate Analysis
TECA is a collection of climate analysis algorithms geared toward extreme event detection and tracking implemented in a scalable parallel framework. The code has been successfully deployed and run at massive scales on current DOE supercomputers. TECA's core is written in modern C++ and exploits MPI + X parallelism where X is one of threads, OpenMP, or GPUs. The framework supports a number of parallel design patterns including distributed data parallelism and map-reduce. While modern C++ delivers the highest performance, Python bindings make the code approachable and easy to use.

### Documentation
The [TECA User's Guide](https://teca.readthedocs.io/en/latest/) is the authorotative source for documentation on topics such as [installing TECA](https://teca.readthedocs.io/en/latest/installation.html), running TECA's [command line applications](https://teca.readthedocs.io/en/latest/applications.html), and [Python development](https://teca.readthedocs.io/en/latest/python.html). The TECA source code is documented on our [Doxygen site](https://teca.readthedocs.io/en/latest/doxygen/index.html).

### Tutorials
The [TECA tutorials](https://sourceforge.net/p/teca/TECA_tutorials) subversion repository contains slides from previous tutorials.

### Examples
The [TECA examples](https://github.com/LBL-EESA/TECA_examples) repository contains batch scripts and codes illustrating the use of TECA at scale.

### Python
The [TECA Python package]() is available on PyPi or by installing from sources. For more information see the [TECA User's Guide](https://teca.readthedocs.io/en/latest/) sections on [installing TECA](https://teca.readthedocs.io/en/latest/installation.html) and [Python development](https://teca.readthedocs.io/en/latest/python.html).

### CI and Testing
For the latest regression suite results see the [TECA CDash project site](https://my.cdash.org/index.php?project=TECA).

## Copyright Notice
TECA, Copyright (c) 2015, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Innovation & Partnerships Office at  IPO@lbl.gov.

NOTICE.  This software is owned by the U.S. Department of Energy.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, prepare derivative works, and perform publicly and display publicly.  Beginning five (5) years after the date permission to assert copyright is obtained from the U.S. Department of Energy, and subject to any subsequent five (5) year renewals, the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
