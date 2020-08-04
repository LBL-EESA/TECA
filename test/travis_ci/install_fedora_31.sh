#!/bin/bash
set -x

# suck in package lists
dnf update -qq -y

# install deps
# use PIP for Python packages
dnf install -qq -y environment-modules which git-all gcc-c++ gcc-gfortran \
    make cmake swig mpich-devel boost-devel python3-devel python3-pip subversion \
    udunits2 udunits2-devel zlib-devel openssl-devel wget redhat-rpm-config

if [[ ${REQUIRE_NETCDF_MPI} == TRUE ]]
then
    dnf install -qq -y  hdf5-mpich-devel netcdf-mpich-devel
else
    dnf install -qq -y hdf5-devel netcdf-devel
fi

set +x
source /usr/share/Modules/init/bash
module load mpi/mpich-x86_64
set -x

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
