Installation
============
TECA is designed to deliver the highest available performance on platforms
ranging from Cray supercomputers to laptops. The installation procedure depends
on the platform and desired use.

.. _install_hpc:

On a Cray Supercomputer
-----------------------
When installing TECA on a supercomputer one of the best options is the
superbuild, a piece of CMake code that downloads and builds TECA and its many
dependencies. The superbuild is located in a git repository here
TECA_superbuild_.

.. _TECA_superbuild: https://github.com/LBL-EESA/TECA_superbuild

Installing TECA on the Cray requires pointing the superbuild to Cray's MPI and
is sensitive to the modules loaded at compile time. You want a clean
environment with only GNU compilers and CMake modules loaded. The superbuild
will take care of compiling and installing TECA and its dependencies. As
occasionally occurs, something may go awry. If that happens it is best to start
over from scratch by `rm -rf *` the contents of both build and install
directories. This is because of CMake's caching mechanism which will remember
bad values and also because the superbuild makes use of what's already been
installed as it progresses and if you don't clean it out you may end up
referring to a broken library.

Overview
~~~~~~~~
A quick overview of installing the latest stable version of TECA from the
`master` branch is:

1.  module swap PrgEnv-intel PrgEnv-gnu
2.  module load cmake
3.  git clone https://github.com/LBL-EESA/TECA_superbuild.git
4.  cd TECA_superbuild
5.  git checkout master
6.  mkdir build && cd build
7.  run `config-teca-sb.sh ..`

When using the resulting TECA install you'll need to load the teca modulefile
installed by the superbuild find this file in the `modulefiles` directory.

.. _sb_version:

Versioning
~~~~~~~~~~
Over time both TECA and its dependencies evolve such that the compatible
versions of dependencies and build options change. This is handled by tagging
releases in both TECA and the TECA_superbuild. Each such stable release of TECA
is paired with a corresponding release of the TECA_superbuild. The superbuild
must be explicitly checked out to the same TECA release one is compiling. Be
sure to call `git checkout X` in the superbuild clone where `X` is either
`master`, `develop`, or a release number found at the `releases`_  page of the
TECA github repo.

Here we show how to install from the `master` branch which has the latest
stable version of the code.  One may also wish to install from the `develop`
branch which gets one the most up to date experimental codes. No matter which
branch or release is selected the process is similar, one will modify the
examples by replacing `master` with `develop` or the desired release number.

.. _releases: https://github.com/LBL-EESA/TECA/releases

.. _fs_selection:

Selecting the file system
~~~~~~~~~~~~~~~~~~~~~~~~~
Supercomputers often come with a number of file systems attached. The choice of
which file system to install the software on can impact the performance of
large scale runs as many processes simultaneously will load code from library
files stored on disk. Often home directories are located on a low performance
networked file system not designed for use at large parallel concurrencies making
the home folder a poor choice for an install. Typically there is also a high
throughput parallel file system, such as Lustre, available that is designed for
massive concurrent access.  Parallel file systems will deliver the best
performance and can be a good choice for installs. Consult your HPC center's
documentation before deciding on a file system for the install. For more information
on the file systems available at NERSC see :ref:`nersc_file_systems`.

Configure script
~~~~~~~~~~~~~~~~
The configure script is a shell script that captures build settings specific to
Cray environment (see step 7 above). You may wish to edit the `TECA_SOURCE` and
`TECA_INSTALL` variables before running the script. Additional CMake arguments
may be passed on the command line. You must pass the location of the superbuild
sources(see step 3 above).

Here is the script `config-teca-sb.sh` in use at NERSC for Cori.

.. code-block:: bash

    #!/bin/bash

    # use the GNU compiler collection
    module swap PrgEnv-intel PrgEnv-gnu

    # mpich is not in the pkg-config path on Cori
    export PKG_CONFIG_PATH=$CRAY_MPICH_DIR/lib/pkgconfig:$PKG_CONFIG_PATH

    # set the banch of TECA and the path where TECA is installed to.
    TECA_SOURCE=master
    TECA_INSTALL=$SCRATCH/teca_installs/$TECA_SOURCE

    echo TECA_SOURCE=${TECA_SOURCE}
    echo TECA_INSTALL=${TECA_INSTALL}
    rm -rfI ${TECA_INSTALL}

    # Configure TECA superbuild
    cmake \
      -DCMAKE_CXX_COMPILER=`which g++` \
      -DCMAKE_C_COMPILER=`which gcc` \
      -DCMAKE_BUILD_TYPE=Release \
      -DTECA_SOURCE=${TECA_SOURCE} \
      -DCMAKE_INSTALL_PREFIX=${TECA_INSTALL} \
      -DENABLE_MPICH=OFF \
      -DENABLE_OPENMPI=OFF \
      -DENABLE_CRAY_MPICH=ON \
      -DENABLE_TECA_TEST=OFF \
      -DENABLE_TECA_DATA=ON \
      $*

    # build and install
    make -j16 install

This script configures the superbuild such that the release or branch named in
`TECA_SOURCE` variable is compiled and installed in the `TECA_INSTALL`
directory.  As discussed in :ref:`sb_version`, the superbuild should be checked
out to the same branch or release number  named in `TECA_source`.  Note that
`$SCRATCH` is a NERSC specific environment variable that points to a directory
owned by the user on a Lustre based file system. On other HPC centers you will
need to replace `$SCRATCH` with the path to the file system you wish to install
TECA on. See :ref:`fs_selection` for more information.

.. hint::

   When trouble shooting the superbuild it is necessary to `rm -rf`
   both the build and install prefix directories. Failing to do so will lead to
   confusing build failures.

Configuring the runtime environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During the install an environment module is generated and installed to
`$TECA_INSTALL/modulefiles/`. To use the new install of TECA you will need to
use it to configure the run time environment.

.. code-block:: bash

    module swap PrgEnv-intel PrgEnv-gnu
    module use ${TECA_INSTALL}/modulefiles
    module load teca

The teca module must be loaded each time you use TECA and is usually best done from
within your batch script.

Debugging and development on a supercomputer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Modifying the source code directly in the superbuild is a cumbersome process.
It is far easier to keep a separate build for development and debugging.  In
this case the superbuild is still useful for installing the dependencies.  To
setup for TECA development and debugging on a supercomputer run the superbuild
with `-DENABLE_TECA=OFF`. This will build the dependencies but not TECA itself.
Once the superbuild completes, load the installed module, and compile a
separate clone of the TECA github repo (See :ref:`compile`). This enables one
to make local modifications, quickly recompile, and run out of the build
directory.

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
directly to CMake on the command line. While not recommended, as a last resort
one may disable a problematic dependency using `-DREQUIRE_<X>=OFF` where X is
the dependency.

.. _install-deps:

Installing dependencies
~~~~~~~~~~~~~~~~~~~~~~~
Most of the dependencies can be installed by the OS specific package manager.
For Python package dependencies pip is used as described in :ref:`python-environment`.

It is recommended to have a parallel HDF5 based NetCDF install, on some systems
(Ubuntu, Mac) this requires installing NetCDF from source as outlined in
:ref:`netcdf-parallel-4`.

Apple Mac OS
^^^^^^^^^^^^

.. code-block:: bash

    brew update
    brew unlink python@2
    brew install netcdf mpich swig svn udunits openssl python

Ubuntu 20.04
^^^^^^^^^^^^

.. code-block:: bash

    $ apt-get update
    $ apt-get install -y gcc g++ gfortran cmake swig \
        libmpich-dev libhdf5-dev libnetcdf-dev \
        libboost-program-options-dev python3-dev python3-pip \
        libudunits2-0 libudunits2-dev zlib1g-dev libssl-dev

Fedora 32
^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^
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
~~~~~~~~~~~~
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

.. _py-only-install:

Python only
-----------
TECA's C++ codes are wrapped in Python, and a number of pure Python
implementations exist in the code base as well. This makes it possible to
develop new TECA applications in Python using the teca Python package.  Two
installation methods have been documented here, `pip` and `conda`.  Currently
the `conda` method has some limitations. As a result `pip` is the recommended
method.

.. _pip_install:

with pip
~~~~~~~~
The TECA Python package can be installed from PyPi using pip. This may be
useful for developing new Python based applications and post processing codes.
A virtual environment is recommended.

Before attempting to install TECA, install system library dependencies as shown
in section :ref:`install-deps`. Pure Python package dependencies may then be
installed via pip.

.. code-block:: bash

   python3 -m venv py3k-teca
   source py3k-teca/bin/activate
   pip3 install numpy matploptlib mpi4py torch
   pip3 install teca

.. note::

    When installing PyTorch, especially when using GPUs, follow the
    `PyTorch_install` instructions found on the PyTorch site.

.. _PyTorch_install: https://pytorch.org/get-started/locally/#start-locally

The `pip install teca` command may take a few minutes as TECA compiles from
sources. Errors are typically due to missing dependencies, from the
corresponding CMake output it should be apparent which dependency was not
found.

TECA makes heavy use of MPI and NetCDF parallel I/O. On some systems, notably
Unbuntu and Mac OS the MPI enabled NetCDF libraries available from package
managers are broken or missing. In this case one can install NetCDF with MPI
features enabled (in NetCDF docs this is called "parallel 4") and point the
build to the local install by passing options on the pip command line.

.. code-block:: bash

   pip install teca --global-option=build_ext \
       --global-option="--with-netcdf=/Users/bloring/netcdf-c-4.7.4-install/"

See section :ref:`netcdf-parallel-4` for information on compiling NetCDF with
MPI enabled.

with conda
~~~~~~~~~~
The following is an experimental recipe for installing TECA into a conda environment.

.. code-block:: bash

   conda create --yes -c conda-forge -n tecapy \
       python=3.9 numpy mpi4py netCDF4 boost openmpi \
       matplotlib python-dateutil cython swig pyparsing \
       cycler pytz torch
   source activate tecapy
   pip install teca --global-option=build_ext \
       --global-option="--without-netcdf-mpi"

This method does not support parallel I/O. As a result it is recommended to use
:ref:`pip_install` installation method.
