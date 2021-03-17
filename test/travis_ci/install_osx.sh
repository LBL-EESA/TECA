#!/bin/bash
set -x

# override system install
export PATH=/usr/local/bin:$PATH

# install deps. note that many are included as a part of brew-core
# these days. hence this list isn't comprehensive
brew update
brew upgrade
brew unlink python@2
brew install openmpi swig svn udunits openssl python@3.8 curl cmake
brew unlink python
brew link --force python@3.8
brew link curl --force

# matplotlib currently doesn't have a formula
# teca fails to locate mpi4py installed from brew
python3 -mvenv `pwd`/../tci
set +x
source `pwd`/../tci/bin/activate
set -x
pip3 install numpy mpi4py matplotlib torch

# install data files.
# On Apple svn is very very slow. On my mac book pro
# this command takes over 14 minutes, while on a Linux
# system on the same network it takes less than 2 minutes.
# travis will kill a build  if it does not get console output
# for 10 min. The following snippet sends progress marks
# to the console while svn runs.
svn co svn://svn.code.sf.net/p/teca/TECA_data@${TECA_DATA_REVISION} TECA_data &
svn_pid=$!

set +x
while [ -n "$(ps -p $svn_pid -o pid=)" ]
do
  echo -n "."
  sleep 2s
done
