#!/bin/bash

# override system install
export PATH=/usr/local/bin:$PATH

# install deps
brew update
brew tap Homebrew/homebrew-science
brew install gcc openmpi hdf5 netcdf python python3 swig svn udunits
# brew install boost --c++11
pip install numpy mpi4py matplotlib
pip3 install numpy mpi4py matplotlib

# install cmake manually because brew version too old
wget https://cmake.org/files/v3.5/cmake-3.5.2-Darwin-x86_64.tar.gz
tar -C /usr/local -x -z -f cmake-3.5.2-Darwin-x86_64.tar.gz --strip-components=3

# install data files.
# On Apple svn is very very slow. On my mac book pro
# this command takes over 14 minutes, while on a Linux
# system on the same network it takes less than 2 minutes.
# travis will kill a build  if it does not get console output
# for 10 min. The following snippet sends progress marks
# to the console while svn runs.
echo 'svn co svn://missmarple.lbl.gov/work3/teca/TECA_data@${TECA_DATA_REVISION} &'
svn co svn://missmarple.lbl.gov/work3/teca/TECA_data@${TECA_DATA_REVISION} &
svn_pid=$!
while [ -n "$(ps -p $svn_pid -o pid=)" ]
do
  echo -n "."
  sleep 2s
done
