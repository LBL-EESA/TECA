# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04


ARG mpich=4.0.2
ARG hdf5=1.14.4-3
ARG hdf5_dot=1.14.4.3
ARG netcdf=4.9.2
ARG INSTALL_PREFIX=/usr/local
ARG MAKE_J=16

# Set non-interactive frontend (avoid prompts from apt)
ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"

# Install Python and other dependencies
RUN apt-get update -qq && apt-get install -qq -y \
    python3 python3-pip \
    git-core gcc g++ gfortran cmake swig \
    libboost-program-options-dev libcurl4-openssl-dev python3-dev \
    subversion libudunits2-0 libudunits2-dev zlib1g-dev libssl-dev m4 wget \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# install mpich manually
ARG mpich_prefix=mpich-$mpich
ARG mpich_fflags=-fallow-argument-mismatch
RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    FFLAGS=$mpich_fflags FCFLAGS=$mpich_fflags ./configure                     \
        --prefix=$INSTALL_PREFIX                                            && \
    make -j $MAKE_J install                                                 && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix $mpich_prefix.tar.gz
RUN /sbin/ldconfig

# install hdf5 against mpich
ARG hdf5_prefix=hdf5-$hdf5
ARG hdf5_flags="--enable-parallel"
RUN \
    wget https://github.com/HDFGroup/hdf5/releases/download/hdf5_$hdf5_dot/hdf5-$hdf5.tar.gz && \
    tar -xvzf hdf5-$hdf5.tar.gz                                                              && \
    cd $hdf5_prefix                                                                          && \
    ./configure CC=mpicc CXX=mpicxx --enable-build-mode=production                              \
        --enable-parallel --enable-shared --disable-static --prefix=$INSTALL_PREFIX          && \
    make -j $MAKE_J install                                                                  && \
    make install                                                                             && \
    cd ..                                                                                    && \
    rm -rf $hdf5_prefix $hdf5_prefix.tar.gz

# install netcdf manually against hdf5
ARG netcdf_prefix=netcdf-c-$netcdf
RUN wget https://downloads.unidata.ucar.edu/netcdf-c/$netcdf/netcdf-c-$netcdf.tar.gz         && \
    tar -xzvf $netcdf_prefix.tar.gz                                                          && \
    cd $netcdf_prefix                                                                        && \
    CC=mpicc ./configure --enable-shared --disable-static --disable-dap                         \
        --enable-netcdf-4 --enable-parallel4 --disable-libxml2 --prefix=$INSTALL_PREFIX      && \
    make -j 16 install && \
    cd .. && \
    rm -rf $netcdf_prefix $netcdf_prefix.tar.gz

# Setup virtual environment and install Python packages
RUN pip3 install --break-system-packages \
    "numpy<2.0" \
    mpi4py \
    matplotlib