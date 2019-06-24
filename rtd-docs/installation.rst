
************
Installation
************

*This guide was tested on a Docker container*

Installing TECA from Sources
############################

Before starting
===============

Clone the project

::

    $ git clone https://github.com/LBL-EESA/TECA.git

In this guide we'll set few env variables for clarification:

.. csv-table:: Env Variables
   :header: "Variable Name", "Function"
   :widths: 20, 60

   "$TECA_DIR", "points to the TECA project we just cloned"
   "$TECA_BUILD_DIR", "the directory that we'll build TECA at"
   "$TECA_INSTALL_DIR", "the directory that we'll install TECA at"

For example::

    $ export TECA_DIR=/project/TECA
    $ export TECA_BUILD_DIR=/project/TECA/build
    $ export TECA_INSTALL_DIR=/project/teca_install

Installing dependencies
=======================

*So far TECA has been tested on Ubuntu 18.04 & Fedora 28*

Installing linux Packages
-------------------------

Install:

* C/C++ & Fortran Compilers 
* CMake
* MPI
* HDF5
* NetCDF
* Boost
* Python & pip
* UDUNITS
* zlib
* OpenSSL

Ubuntu 18.04
~~~~~~~~~~~~

::

    $ apt-get update
    $ apt-get install -y gcc g++ gfortran cmake swig \
        libmpich-dev libhdf5-dev libnetcdf-dev \
        libboost-program-options-dev python3-dev python3-pip \
        libudunits2-0 libudunits2-dev zlib1g-dev libssl-dev

Fedora 28
~~~~~~~~~

::

    $ dnf update
    $ dnf install -y gcc-c++ gcc-gfortran make cmake-3.11.0-1.fc28 \
        swig mpich-devel hdf5-devel netcdf-devel boost-devel \
        python3-devel python3-pip udunits2-devel zlib-devel openssl-devel

Don't forget to load the ``MPI`` module on Fedora::

    $ module load mpi

Installing Python Packages
--------------------------

* Numpy
* mpi4py
* Matplotlib

::

    $ pip3 install numpy mpi4py matplotlib

For Matplotlib, choose `AGG backend <https://matplotlib.org/3.1.0/tutorials/introductory/usage.html#what-is-a-backend>`_::

    $ export MPLBACKEND=Agg


Installation Step
=================

There are 2 ways to install TECA. It's either by running ``setup.py`` or ``cmake``.

setup.py
--------

::

    $ cd $TECA_DIR
    $ python3 setup.py install

cmake
-----

create the ``build`` dir and ``cd`` to it

::

    $ mkdir $TECA_BUILD_DIR
    $ cd $TECA_BUILD_DIR

And finally install TECA

::

    $ cmake -DCMAKE_INSTALL_PREFIX=$TECA_INSTALL_DIR $TECA_DIR
    $ make && make install

Setting env variables to include TECA
=====================================

::

    $ export PATH=${TECA_INSTALL_DIR}/bin:${PATH}
    $ export PYTHONPATH=${TECA_INSTALL_DIR}/lib:${PYTHONPATH}
    $ export LD_LIBRARY_PATH=${TECA_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

If everything went well we can confirm that it worked by trying::

    $ python3 -c "from teca import *"

Now we're ready to go!
