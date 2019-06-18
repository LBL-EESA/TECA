
************
Installation
************

*This guide was tested on a Docker container*

Ubuntu
######

In this guide we'll assume the working directory is in env variable ``$TECA_DIR``::

    $ export $TECA_DIR=/path/to/TECA

Installing dependencies
=======================

Update::

    $ apt-get update

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

::

    $ apt-get install -y gcc g++ gfortran cmake swig \
        libmpich-dev libhdf5-dev libnetcdf-dev \
        libboost-program-options-dev python3-dev python3-pip \
        libudunits2-0 libudunits2-dev zlib1g-dev libssl-dev

* Libxlsxwriter

::

    $ # Libxlsxwriter: A C library for creating Excel XLSX files.
    $ git clone http://github.com/burlen/libxlsxwriter.git
    
    $ cd libxlsxwriter && make && make install

* Numpy
* mpi4py
* Matplotlib

::

    $ pip3 install numpy mpi4py matplotlib

For Matplotlib, choose `AGG backend <https://matplotlib.org/3.1.0/tutorials/introductory/usage.html#what-is-a-backend>`_::

    $ export MPLBACKEND=Agg

Installing TECA
===============

cd to our working dir

::
    
    $ cd $TECA_DIR

create a building directory

::

    $ mkdir build && cd build

And finally install TECA

::

    $ cmake ..
    $ make && make install

Setting env variables to include TECA

::

    $ export PATH=${TECA_DIR}:${PATH}
    $ export PYTHONPATH=${TECA_DIR}/build/lib:${PYTHONPATH}
    $ export LD_LIBRARY_PATH=${TECA_DIR}/build/lib:${LD_LIBRARY_PATH}

If everything went well we can confirm that it worked by trying::

    $ python3 -c "from teca import *"

Now we're ready to go!






