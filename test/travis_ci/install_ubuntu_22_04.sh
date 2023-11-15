#!/bin/bash
set -x

export DEBIAN_FRONTEND="noninteractive"

# To address tzdata configuration resulted by installing cmake
export TZ="America/Los_Angeles"

# suck in package lists
apt-get update -qq

# install deps
# use PIP for Python packages
apt-get install -qq -y git-core gcc g++ gfortran cmake swig libmpich-dev \
    libboost-program-options-dev python3-dev python3-pip subversion \
    libudunits2-0 libudunits2-dev zlib1g-dev libssl-dev m4 wget

if [[ ${REQUIRE_NETCDF_MPI} == TRUE ]]
then
    apt-get install -qq -y libhdf5-mpich-dev

    wget https://downloads.unidata.ucar.edu/netcdf-c/4.8.1/netcdf-c-4.8.1.tar.gz
    tar -xzvf netcdf-c-4.8.1.tar.gz && cd netcdf-c-4.8.1
    ./configure CC=mpicc CFLAGS="-O2 -g -I/usr/include/hdf5/mpich" \
        LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/mpich/ -lhdf5" \
        --enable-parallel-4 --disable-dap
    make -j install
    cd ..
else
    apt-get install -qq -y libhdf5-dev libnetcdf-dev
fi

echo ${TRAVIS_BRANCH}
echo ${BUILD_TYPE}
echo ${DOCKER_IMAGE}
echo ${IMAGE_VERSION}
echo ${TECA_PYTHON_VERSION}
echo ${TECA_DATA_REVISION}

python3 -mvenv `pwd`/../tci
set +x
source `pwd`/../tci/bin/activate
set -x
pip3 install numpy mpi4py matplotlib torch

# install data files.
svn co svn://svn.code.sf.net/p/teca/TECA_data@${TECA_DATA_REVISION} TECA_data
