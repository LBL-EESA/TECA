Installation
============
TECA is designed to deliver the highest available performance on platforms
ranging from Cray supercomputers to laptops. The installation procedure depends
on the platform and desired use.


On a Cray Supercomputer
-----------------------
When installing TECA on a supercomputer one of the best options is the
superbuild, a piece of CMake code that downloads and builds TECA and its now
more than 26 dependencies. The superbuild is located in a git repository here
TECA_superbuild_.

.. _TECA_superbuild: https://github.com/LBL-EESA/TECA_superbuild

Installing TECA on the Cray requires pointing the superbuild to Cray's MPI and
is sensitive to the modules loaded at compile time. You want a clean
environment with only GNU compilers and CMake modules loaded. The GNU compilers
are C++11 and Fortran 2008 standards compliant, while many other compilers are
not. Additionally the HDF5 HL library needs to be built with thread safety
enabled, which is rarely the case at the time of writing. Finally, some HPC
centers have started to use Anaconda Python which can mix in incompatible
builds of various libraries. The superbuild will take care of compiling and
installing TECA and its dependencies. As occasionally occurs, something may go
awry. If that happens it is best to start over from scratch. Starting over from
scratch entails rm'ing contents of both build and install directories. This is
because of CMake's caching mechanism which will remember bad values and also
because the superbuild makes use of what's already been installed as it
progresses and if you don't clean it out you may end up referring to a broken
library.

Overview
~~~~~~~~
A quick overview follows. Note that step 8 is for using TECA after.
This is usually put in an environment module.

1. module swap PrgEnv-intel PrgEnv-gnu
2. module load cmake/3.8.2
3. git clone https://github.com/LBL-EESA/TECA_superbuild.git
4. mkdir build && cd build
5. edit install paths and potentially update library paths after NERSC upgrades OS in config-teca-sb.sh. (see below)
6. run config-teca-sb.sh .. (you may need to replace .. with path to super build clone)
7. make -j 32 install
8. When using TECA you need to source teca_env.sh from the install's bin dir, load the correct version of gnu programming environment, and ensure that no incompatible modules are loaded(Python, HDF5, NetCDF etc).


Configure script
~~~~~~~~~~~~~~~~
The configure script is a shell script that captures build settings specific to
Cray environment (see steps 5 & 6 above). You will need to edit the TECA_VER
and TECA_INSTALL variables before running it. Additional CMake arguments may be
passed on the command line. You must pass the location of the superbuild
sources(see step 3 above).

Here is the script config-teca-sb.sh use at NERSC for Cori and Edison.

.. code-block:: bash

    #!/bin/bash

    # mpich is not in the pkg-config path on Cori/Edison
    # I have reported this bug to NERSC. for now we
    # must add it ourselves.
    export PKG_CONFIG_PATH=$CRAY_MPICH_DIR/lib/pkgconfig:$PKG_CONFIG_PATH

    # set the the path where TECA is installed to.
    # this must be a writable directory by the user
    # who is doing the build, as install is progressive.
    TECA_VER=2.1.1
    TECA_INSTALL=/global/cscratch1/sd/loring/test_superbuild/teca/$TECA_VER

    # Configure TECA superbuild
    cmake \
      -DCMAKE_CXX_COMPILER=`which g++` \
      -DCMAKE_C_COMPILER=`which gcc` \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${TECA_INSTALL} \
      -DENABLE_MPICH=OFF \
      -DENABLE_OPENMPI=OFF \
      -DENABLE_CRAY_MPICH=ON \
      -DENABLE_TECA_TEST=OFF \
      -DENABLE_TECA_DATA=ON \
      -DENABLE_READLINE=OFF \
      $*


On a laptop or desktop
----------------------
On a laptop or desktop system one may use local package managers to install
third-party dependencies, and then proceed with compiling and installing TECA.
A simple procedure exists for those wishing to use TECA for Python scripting.
See section :ref:`py-only-install`. For those wishing access to TECA
libraries, command line applications, and Python scripting, compiling from
sources is the best option. See section :ref:`compile`.

Note, that as with any install, post install the environment will likely need
to be set to pick up the install.  Specifically, PATH, LD_LIBRARY_PATH (or
DYLD_LIBRRAY_PATH on Mac), and PYTHONPATH need to be set correctly. See section
:ref:`post-install`.

.. _py-only-install:

A Python only install
~~~~~~~~~~~~~~~~~~~~~
It is often convenient to install TECA locally for post processing results from
runs made on a supercomputer. If one only desires access to the Python package,
one may use pip. It is convenient, but not required, to do so in a virtual env.

Before attempting to install TECA, install dependencies as shown in section
:ref:`install-deps`

.. code-block:: bash

   python3 -m venv py3k
   source py3k/bin/activate
   pip3 install numpy matploptlib mpi4py teca

The install may take a few minutes as TECA compiles from sources. Errors are
typically due to missing dependencies, from the corresponding message it should
be obvious which dependency is missing.

.. _compile:

Compiling TECA from sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~
TECA depends on a number of third party libraries. Before attempting to compile
TECA please install dependencies as described in section
:ref:`install-deps` and then set up the Python environment as described in
section :ref:`python-environment`.

Once dependencies are installed, a typical install might proceed as follows.

.. code-block:: bash

   git clone https://github.com/LBL-EESA/TECA.git
   svn co svn://svn.code.sf.net/p/teca/TECA_data TECA_data
   cd TECA
   mkdir bin
   cd bin
   cmake ..
   make -j
   make -j install

If all goes well, at the end of this TECA will be installed. However, note that
the install location should be added to various system paths, See :ref:`post-install`
for how to configure the run time environment.

When running CMake one can pass `-DCMAKE_INSTALL_PREFIX=<some path>` to control
where the install lands, and `-DBUILD_TESTING=ON` to enable regression tests.

The most common problem is when CMake failed to locate a dependency. Usually
the error message has information about correcting the situation. Usually the
remedy is to explicitly pass the path where the dependency is installed
directly to CMake on the command line. While not recommened, as a last resort
one may disable a problematic dependency using `-DREQUIRE_<X>=OFF` where X is
the dependency.

.. _install-deps:

Installing dependencies
~~~~~~~~~~~~~~~~~~~~~~~
Most of the dependencies can be installed by the OS specific package manager.
For Python modules pip is used as described in :ref:`python-environment`.

It is recommended to have a parallel HDF5 based NetCDF install, on some systems
(Ubuntu, Mac) this requires installing NetCDF from source as outlined in
:ref:`netcdf-parallel-4`.

Apple Mac OS
++++++++++++

.. code-block:: bash

    brew update
    brew unlink python@2
    brew install netcdf mpich swig svn udunits openssl python

Ubuntu 20.04
++++++++++++

.. code-block:: bash

    $ apt-get update
    $ apt-get install -y gcc g++ gfortran cmake swig \
        libmpich-dev libhdf5-dev libnetcdf-dev \
        libboost-program-options-dev python3-dev python3-pip \
        libudunits2-0 libudunits2-dev zlib1g-dev libssl-dev

Fedora 32
+++++++++

.. code-block:: bash

    $ dnf update
    $ dnf install -qq -y environment-modules which git-all gcc-c++ gcc-gfortran \
        make cmake swig mpich-devel hdf5-mpich-devel netcdf-mpich-devel \
        boost-devel python3-devel python3-pip subversion udunits2 udunits2-devel \
        zlib-devel openssl-devel wget redhat-rpm-config

Some of these packages may need an environment module loaded, for instance ``MPI``

.. code-block:: bash

    $ module load mpi

.. _python-environment:

Python environment
++++++++++++++++++

TECA's Python dependencies can be easily installed via pip.

.. code-block:: bash
    
    $ pip3 install numpy mpi4py matplotlib torch

However, when building TECA from sources it can be useful to setup a virtual
environment.  Creating the virtual environment is something that you do once,
typically in your home folder or the SCRATCH file system on the Cray. Once
setup the venv will need to be activated each time you use TECA.

.. code-block:: bash

    $ cd ~
    $ python3 -mvenv teca-py3k
    $ source teca-py3k/bin/activate
    $ pip3 install numpy matplotlib mpi4py torch  

Before building TECA, and every time you use TECA be sure to activate the same venv.

.. code-block:: bash

    $ source teca-py3k/bin/activate

Once the venv is installed and activated, see :ref:`compile`.

.. note::

    As of 1/1/2020 TECA switched to Python 3. Python 2 may still work
    but is no longer maintained and should not be used.


.. _netcdf-parallel-4:

NetCDF w/ Parallel 4
+++++++++++++++++++++
As of 7/31/2020 TECA relies on HDF5 NetCDF with MPI collective I/O. The
NetCDF project calls this feature set "parallel 4". At this time neither
Mac OS homebrew nor Ubuntu 20.04 have a functional parallel NetCDF package.
On those systems one should install NetCDF from sources.

On Ubuntu 20.04

.. code-block:: bash

    $ cd ~
    $ sudo apt-get remove libhdf5-dev
    $ sudo apt-get install libmpich-dev libhdf5-mpich-dev
    $ wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-c-4.7.4.tar.gz
    $ tar -xvf netcdf-c-4.7.4.tar.gz
    $ cd netcdf-c-4.7.4
    $ ./configure CC=mpicc CFLAGS="-O3 -I/usr/include/hdf5/mpich"       \
          LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/mpich/ -lhdf5"      \
          --prefix=`pwd`/../netcdf-c-4.7.4-install --enable-parallel-4  \
          --disable-dap
    $ make -j install

On Apple Mac OS

.. code-block:: bash

    $ brew uninstall netcdf hdf5 mpich
    $ brew install mpi hdf5-mpi
    $ wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-c-4.7.4.tar.gz
    $ tar -xvf netcdf-c-4.7.4.tar.gz
    $ cd netcdf-c-4.7.4
    $ ./configure CC=mpicc --enable-shared --enable-static          \
        --enable-fortran --disable-dap --enable-netcdf-4            \
        --enable-parallel4 --prefix=`pwd`/../netcdf-c-4.7.4-install
    $ make -j install


.. _post-install:

Post Install
------------
When installing after compiling from sources the user's environment should be
updated to use the install. One may use the following shell script as a
template for this purpose by replacing @CMAKE_INSTALL_PREFIX@ and
@PYTHON_VERSION@ with the value used during the install.

.. code-block:: bash

    #!/bin/bash

    export LD_LIBRARY_PATH=@CMAKE_INSTALL_PREFIX@/lib/:@CMAKE_INSTALL_PREFIX@/lib64/:$LD_LIBRARY_PATH
    export DYLD_LIBRARY_PATH=@CMAKE_INSTALL_PREFIX@/lib/:@CMAKE_INSTALL_PREFIX@/lib64/:$DYLD_LIBRARY_PATH
    export PKG_CONFIG_PATH=@CMAKE_INSTALL_PREFIX@/lib/pkgconfig:@CMAKE_INSTALL_PREFIX@/lib64/pkgconfig:$PKG_CONFIG_PATH
    export PYTHONPATH=@CMAKE_INSTALL_PREFIX@/lib:@CMAKE_INSTALL_PREFIX@/lib/python@PYTHON_VERSION@/site-packages/
    export PATH=@CMAKE_INSTALL_PREFIX@/bin/:$PATH

    # for server install without graphics capability
    #export MPLBACKEND=Agg

With this shell script in hand one configures the environment for use by sourcing it.

When developing TECA it is common to skip the install step and run out of the
build directory. When doing so one must also set LD_LIBRARY_PATH,
DYLD_LIBRARY_PATH, PYTHONPATH, and PATH to point to the build directory.
