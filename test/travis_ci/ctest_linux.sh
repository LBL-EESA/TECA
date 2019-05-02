#!/bin/bash
set -v
export DASHROOT=`pwd`
export CC=`which gcc-5`
export CXX=`which g++-5`
export FC=`which gfortran-5`
export SHA=`git log --pretty=format:'%h' -n 1`
export PATH=.:${PATH}
export PYTHONPATH=${TRAVIS_BUILD_DIR}/build/lib
export LD_LIBRARY_PATH=${TRAVIS_BUILD_DIR}/build/lib
export MPLBACKEND=Agg
mkdir build
cmake --version
ctest -S ${DASHROOT}/test/travis_ci/ctest_linux.cmake -V
