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
    libopenmpi-dev openmpi-bin libhdf5-openmpi-dev libnetcdf-dev \
    libboost-program-options-dev python-dev

pip install --user numpy mpi4py

# install git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
sudo rm /usr/local/bin/git-lfs

# install data files.
git clone https://github.com/LBL-EESA/TECA_data.git
pushd TECA_data && git lfs pull && popd
