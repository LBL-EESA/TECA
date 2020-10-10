#!/bin/bash
set -v

# setup repo with recent gcc versions
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo add-apt-repository -y ppa:teward/swig3.0
#sudo add-apt-repository -y ppa:george-edison55/cmake-3.x

# suck in package lists
sudo apt-get update -qq

# install deps
# use PIP for Python packages
sudo apt-get install -qq -y gcc-5 g++-5 gfortran-5 swig3.0 \
    libmpich-dev libhdf5-dev libnetcdf-dev libboost-program-options-dev \
    python-dev subversion libudunits2-0 libudunits2-dev

pip install --user numpy mpi4py matplotlib torch

# install cmake manually because repo/ppa versions are too old
wget https://cmake.org/files/v3.5/cmake-3.5.2-Linux-x86_64.tar.gz
sudo tar -C /usr -x -z -f cmake-3.5.2-Linux-x86_64.tar.gz --strip-components=1

# install data files.
svn co svn://svn.code.sf.net/p/teca/TECA_data@${TECA_DATA_REVISION} TECA_data
svn co svn://svn.code.sf.net/p/teca/TECA_assets@${TECA_ASSETS_REVISION} TECA_assets
