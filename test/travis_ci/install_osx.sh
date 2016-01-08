#!/bin/bash

# install deps
brew update
brew tap Homebrew/homebrew-science
brew install gcc openmpi hdf5 netcdf python swig
pip install numpy
# TODO
# brew install boost --c++11

# override system install
export PATH=/usr/local/bin:$PATH
