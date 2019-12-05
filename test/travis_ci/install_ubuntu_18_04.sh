#!/bin/bash
set -v

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

pip${TECA_PYTHON_VERSION} install numpy mpi4py matplotlib

# install data files.
svn co svn://missmarple.lbl.gov/work3/teca/TECA_data@${TECA_DATA_REVISION} TECA_data
