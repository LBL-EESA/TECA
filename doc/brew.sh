#!/bin/bash

# if you are having issues then unistall everything and start from
# scratch
#ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall)"
# you may need to clean up /usr/local/
# you may need to prepend /usr/local/bin to the PATH in ~/.bashrc

# install homebrew
#ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# some basics
#brew install gcc
#brew install gdb

# here are teca dependencies
brew install cmake
brew install python
brew install swig
pip install numpy matplotlib
brew install open-mpi
brew tap homebrew/science
brew install netcdf
brew install boost --c++11
#brew install vtk
