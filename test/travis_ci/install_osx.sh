#!/bin/bash

# install deps
brew update
brew tap Homebrew/homebrew-science
brew install gcc openmpi hdf5 netcdf python swig git-lfs
pip install numpy mpi4py
# TODO
# brew install boost --c++11

# override system install
export PATH=/usr/local/bin:$PATH

# install data files.
git clone https://github.com/LBL-EESA/TECA_data.git
pushd TECA_data && git lfs pull && popd
