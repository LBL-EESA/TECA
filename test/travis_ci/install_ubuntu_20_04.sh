#!/bin/bash
set -x

# suck in package lists
apt-get update -qq

# install deps
# use PIP for Python packages
apt-get install -qq -y git-core gcc g++ gfortran cmake swig \
    libmpich-dev libhdf5-dev libnetcdf-dev libboost-program-options-dev \
    python-dev python-pip python3-dev python3-pip subversion libudunits2-0 \
    libudunits2-dev zlib1g-dev libssl-dev

git clone http://github.com/burlen/libxlsxwriter.git
cd libxlsxwriter
make
make install
cd ..

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
