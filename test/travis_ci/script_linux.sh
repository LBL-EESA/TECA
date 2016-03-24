#!/bin/bash
set -v

# setup our python paths
export PYTHONPATH=${TRAVIS_BUILD_DIR}/build/lib
export LD_LIBRARY_PATH=${TRAVIS_BUILD_DIR}/build/lib

mkdir build
cd build

cmake \
    -DCMAKE_C_COMPILER=`which gcc-5` \
    -DCMAKE_CXX_COMPILER=`which g++-5` \
    -DCMAKE_CXX_FLAGS="-Wall -Wextra" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DBUILD_TESTING=ON \
    -DTECA_DATA_ROOT=${TRAVIS_BUILD_DIR}/TECA_data \
    -DTECA_USE_ASAN=ON \
    ..

make -j4
ctest --output-on-failure
