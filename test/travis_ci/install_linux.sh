#!/bin/bash
set -v

# setup repo with recent gcc versions
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo add-apt-repository -y ppa:teward/swig3.0

# suck in package lists
sudo apt-get update -qq

# install deps
# use PIP for Python packages
sudo apt-get install -qq -y cmake gcc-5 g++-5 gfortran swig3.0 \
    libmpich-dev mpich libhdf5-mpich-dev libnetcdf-dev \
    libboost-program-options-dev python-dev subversion

pip install --user numpy mpi4py

# install data files.
svn co svn://missmarple.lbl.gov/work3/teca/TECA_data TECA_data 
