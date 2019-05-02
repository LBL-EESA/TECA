#!/bin/bash
set -v

# setup repo with recent gcc versions
add-apt-repository -y ppa:ubuntu-toolchain-r/test
add-apt-repository -y ppa:teward/swig3.0
#add-apt-repository -y ppa:george-edison55/cmake-3.x

# suck in package lists
apt-get update -qq

# install deps
# use PIP for Python packages
apt-get install -qq -y git-core gcc-5 g++-5 gfortran-5 cmake swig3.0 \
    libmpich-dev libhdf5-dev libnetcdf-dev libboost-program-options-dev \
    python-dev python-pip subversion libudunits2-0 libudunits2-dev \
    zlib1g-dev libssl-dev 

git clone http://github.com/burlen/libxlsxwriter.git
cd libxlsxwriter
make
make install
cd ..

pip install --user numpy mpi4py matplotlib

# install cmake manually because repo/ppa versions are too old
#wget https://cmake.org/files/v3.5/cmake-3.5.2-Linux-x86_64.tar.gz
#tar -C /usr -x -z -f cmake-3.5.2-Linux-x86_64.tar.gz --strip-components=1

# install data files.
svn co svn://missmarple.lbl.gov/work3/teca/TECA_data TECA_data
