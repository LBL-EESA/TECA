#!/bin/bash

# first install homebrew if you don't have it
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# now time to fix any and all error reports
# especially those relating to command line
# tools and Python
brew doctor

# install some basics
brew install gcc
brew install gdb

# install TECA's dependencies
brew install cmake
brew install python
brew install swig
pip install numpy matplotlib
brew install open-mpi
brew tap homebrew/science
brew install netcdf
brew install boost --c++11
#brew install vtk

# if you are having issues at this point
# then unistall everything and start from
# scratch
#ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall)"
# you may need to clean up /usr/local/
# you may need to prepend /usr/local/bin to the PATH in ~/.bashrc

