#!/bin/bash
set -v

# suck in package lists
dnf update -qq -y

# install deps
# use PIP for Python packages
dnf install -qq -y environment-modules which git-all gcc-c++ gcc-gfortran \
    make cmake-3.11.0-1.fc28 swig mpich-devel hdf5-devel netcdf-devel boost-devel \
    python-devel python-pip subversion udunits2 udunits2-devel \
    zlib-devel openssl-devel wget

git clone https://github.com/jmcnamara/libxlsxwriter.git
cd libxlsxwriter
make
make install
cd ..

source /usr/share/Modules/init/bash

module load mpi

pip install --user numpy mpi4py matplotlib

# install data files.
svn co svn://missmarple.lbl.gov/work3/teca/TECA_data TECA_data
