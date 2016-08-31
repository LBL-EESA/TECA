#!/bin/bash

# override system install
export PATH=/usr/local/bin:$PATH

# install deps
brew update
brew tap Homebrew/homebrew-science
brew install gcc openmpi hdf5 netcdf python swig svn udunits
# brew install boost --c++11
pip install numpy mpi4py matplotlib

# install cmake manually because brew version too old
wget https://cmake.org/files/v3.5/cmake-3.5.2-Darwin-x86_64.tar.gz
tar -C /usr/local -x -z -f cmake-3.5.2-Darwin-x86_64.tar.gz --strip-components=3

# install data files.
svn co svn://missmarple.lbl.gov/work3/teca/TECA_data
