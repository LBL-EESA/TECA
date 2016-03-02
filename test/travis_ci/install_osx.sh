#!/bin/bash

# install deps
brew update
brew tap Homebrew/homebrew-science
brew install gcc openmpi hdf5 netcdf python swig svn udunits
pip install numpy mpi4py
# TODO
# brew install boost --c++11

# override system install
export PATH=/usr/local/bin:$PATH

# install data files.
svn co svn://missmarple.lbl.gov/work3/teca/TECA_data
