# The TECA, Toolkit for Extreme Climate Analaysis
TECA(Toolkit for Extreme Climate Analysis). A collection of climate analysis algorithms geared toward extreme event detection and tracking implemented in a scalable parallel framework. The core is written in modern c++ and uses MPI+thread for parallelism. The framework supports a number of parallel design patterns including distributed data parallelism and map-reduce. Python bindings make the high performance c++ code easy to use. TECA has been used up to 750k cores.

## Build and Install from Sources
Building TECA requires a C++11 compiler and CMake. On Unix like systems
GCC 4.9(or newer), or LLVM 3.5(or newer) are capable of compiling TECA.
Additionally, TECA relies on a number of third party libraries for various
features and functionality. The dependencies are all optional in the sense
that the build will proceed if they are missing, however core functionality
may be missing. We highly recommend building TECA with NetCDF, UDUNITS,
MPI, Boost, and Python.


### Installing Dependencies
#### List of Dependencies
The full list of dependencies, not including a compilaer and CMake, are:
* NetCDF 4: Required for CF-2.0 file I/O
* UDUNITS 2: Required for calendaring
* MPI 3: Required for MPI parallel operation
* Python, SWIG 3, NumPy: Required for Python bindings
* mpi4py: Required for parallel Python programming
* Boost: Required for command line c++ applications
* libxlsxwriter: Required for binary MS Excel workbook output
* VTK 6: Required for mesh based file output

#### Ubuntu 14.04
On Ubunut the apt package manager is recommended.
```bash
# setup repo with recent package versions
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo add-apt-repository -y ppa:teward/swig3.0
sudo apt-get update -qq
# install deps
sudo apt-get install -qq -y cmake gcc-5 g++-5 gfortran swig3.0 \
    libopenmpi-dev openmpi-bin libhdf5-openmpi-dev libnetcdf-dev \
    libboost-program-options-dev python-dev
# use PIP for Python packages
pip install --user numpy mpi4py
```
More recent releases of Ubuntu will not need to use PPA repos to obtain up to date packages.
Other distros, such as Fedora, have a similar install procedure albeit with different
package manager and package names.

#### Apple Mac OSX Yosemite
On Apple Mac OSX using homebrew to install the dependencies is recommended.
```bash
brew update
brew tap Homebrew/homebrew-science
brew install cmake gcc openmpi hdf5 netcdf python swig git-lfs
brew install boost --c++11
pip install numpy mpi4py
```
We highly recommend taking a look a the output of \textbf{brew doctor} and
fixing all reported issues before attempting a TECA build. We've found that significant
complications can arise where user's have mixed installation methods, such as mixing
installs from macports, homebrew, or manual installs. Multiple Python installations
can also be problematic. During configuration TECA reports the Python version detected,
one should verify that this is correct and if not set the paths manually.

### Obtaining the TECA Sources and Test Data
To obtain the sources clone our github repository.
```bash
git clone git@github.com:LBL-EESA/TECA.git
```
TECA comes with a suite of regression tests. If you wish to validate your build, you'll
also need to obtain the test datasets.
```bash
svn co svn://missmarple.lbl.gov/work3/teca/TECA_data
```

### Compiling TECA
The following sections show operating specific examples of compiling TECA. In these examples
it is assumed that you have previously installed third party dependencies, cloned the TECA source
code, and data. The full path to the TECA sources on your system should  be
given by `${TECA_SOURCE_DIR}` while the full path to the test data is given by
`${TECA_DATA_DIR}`. Please update these paths accordingly.

\noindent TECA requires an out of source build. The first step is to create a build directory and
cd into it.
```bash
mkdir ${TECA_SOURCE_DIR}/build
cd ${TECA_SOURCE_DIR}/build
```
#### Ubuntu 14.04
```bash
# configure
cmake \
    -DCMAKE_C_COMPILER=`which gcc-5` \
    -DCMAKE_CXX_COMPILER=`which g++-5` \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DTECA_DATA_ROOT=${TECA_DATA_DIR} \
    ${TECA_SOURCE_DIR}
# compile
make -j4
```

#### Apple Mac OSX Yosemite
```bash
# configure
cmake \
    -DCMAKE_C_COMPILER=`which $CC` \
    -DCMAKE_CXX_COMPILER=`which $CXX` \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DTECA_DATA_ROOT=${TECA_DATA_DIR} \
    ${TECA_SOURCE_DIR}
# compile
make -j4
```

### Configuring the Environment
In order to use TECA's Python modules one must set the library and Python paths.
#### Ununtu 14.04
```bash
export PYTHONPATH=${TECA_SOURCE_DIR}/build/lib
export LD_LIBRARY_PATH=${TECA_SOURCE_DIR}/build/lib
```

#### Apple Mac OSX Yosemite
```bash
export PYTHONPATH=${TECA_SOURCE_DIR}/build/lib
export LD_LIBRARY_PATH=${TECA_SOURCE_DIR}/build/lib
export DYLD_LIBRARY_PATH=${TECA_SOURCE_DIR}/build/lib
```

### Validating the Build
TECA comes with an extensive regression test suite. We recommend that you validate
your build by running the regression tests.
```bash
ctest --output-on-failure
```

#Copyright Notice#
TECA, Copyright (c) 2015, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Innovation & Partnerships Office at  IPO@lbl.gov.

NOTICE.  This software is owned by the U.S. Department of Energy.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, prepare derivative works, and perform publicly and display publicly.  Beginning five (5) years after the date permission to assert copyright is obtained from the U.S. Department of Energy, and subject to any subsequent five (5) year renewals, the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
