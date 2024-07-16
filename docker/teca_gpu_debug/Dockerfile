# Use TECA's prerequisites to build a Docker image for TECA's CPU version
FROM obrientaiu/teca_prereqs:gpu

# *********************
# set build parameters
# *********************
#ARG TECA_BRANCH=develop
#ARG CMAKE_BUILD_TYPE=Release
ARG TECA_BRANCH=ctest_fix
ARG CMAKE_BUILD_TYPE=Debug

ARG HOME=/home/$USER
ARG INSTALL_PREFIX=/usr/local
ARG TECA_INSTALL_PREFIX=$INSTALL_PREFIX
ARG TECA_BUILD_ROOT=/TECA_build
ARG TECA_SRC_ROOT=/TECA
ENV TECA_DATA_DIR=/TECA_data
ARG TECA_HYPERTHREADS_PER_CORE=2
ARG REQUIRE_NETCDF_MPI=TRUE

# pull down the latest version of TECA
RUN git clone --recursive https://github.com/LBL-EESA/TECA.git $TECA_SRC_ROOT && \
    cd $TECA_SRC_ROOT && git checkout $TECA_BRANCH && \
    mkdir -p $TECA_BUILD_ROOT && cd $TECA_BUILD_ROOT && \
    cmake \
        -Wno-dev \
        -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
        -DENABLE_MPI=ON \
        -DBUILD_TESTING=ON \
        -DTECA_ENABLE_PROFILER=ON \
        -DTECA_DATA_ROOT=$TECA_DATA_DIR \
        -DTECA_TEST_CORES=16 \
        -DHYPERTHREADS_PER_CORE=$TECA_HYPERTHREADS_PER_CORE \
        -DREQUIRE_OPENSSL=TRUE \
        -DREQUIRE_BOOST=TRUE \
        -DREQUIRE_NETCDF=TRUE \
        -DREQUIRE_NETCDF_MPI=$REQUIRE_NETCDF_MPI \
        -DREQUIRE_UDUNITS=TRUE \
        -DREQUIRE_MPI=TRUE \
        -DREQUIRE_PYTHON=TRUE \
        -DREQUIRE_TECA_DATA=FALSE \
        -DREQUIRE_CUDA=TRUE \
        -DCMAKE_INSTALL_PREFIX=$TECA_INSTALL_PREFIX \
        $TECA_SRC_ROOT && \
    make -j 16 install && \
    cp -r $TECA_BUILD_ROOT/bin /tmp && cp -r $TECA_BUILD_ROOT/lib /tmp && \
    make clean && rm -rf $TECA_BUILD_ROOT/bin && rm -rf $TECA_BUILD_ROOT/lib && \
    mv /tmp/bin $TECA_BUILD_ROOT/ && mv /tmp/lib $TECA_BUILD_ROOT/

# make sure TECA binaries and python packages are accessible
ENV LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib64:$TECA_INSTALL_PREFIX/lib/:$TECA_INSTALL_PREFIX/lib64/:$LD_LIBRARY_PATH \
    PKG_CONFIG_PATH=$INSTALL_PREFIX/lib/pkgconfig:$INSTALL_PREFIX/lib64/pkgconfig:$TECA_INSTALL_PREFIX/lib/pkgconfig:$TECA_INSTALL_PREFIX/lib64/pkgconfig \
    PYTHONPATH=$TECA_INSTALL_PREFIX/lib \
    PATH=$INSTALL_PREFIX/bin/:$TECA_INSTALL_PREFIX/bin/:$TECA_BUILD_ROOT/bin/test/:$PATH
