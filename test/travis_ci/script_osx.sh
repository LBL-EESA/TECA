#!/bin/bash
set -v

# setup our python paths
export PYTHONPATH=${TRAVIS_BUILD_DIR}/build/lib
export LD_LIBRARY_PATH=${TRAVIS_BUILD_DIR}/build/lib
export DYLD_LIBRARY_PATH=${TRAVIS_BUILD_DIR}/build/lib

mkdir build
cd build

cmake \
    -DCMAKE_C_COMPILER=`which $CC` \
    -DCMAKE_CXX_COMPILER=`which $CXX` \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_FLAGS="-Wall -Wextra" \
    -DBUILD_TESTING=ON \
    -DTECA_DATA_ROOT=${TRAVIS_BUILD_DIR}/TECA_data \
    -DTECA_USE_ASAN=ON \
    ..

make -j4
ctest --output-on-failure
