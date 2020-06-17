FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y git-core gcc g++ gfortran cmake \
        python3-dev subversion pkg-config wget && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# MPI for Shifter
ARG mpich=3.3
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure                                                             && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

RUN /sbin/ldconfig

#### CREATE WORKING DIRECTORY FOR USER ####
ARG WORKDIR=/teca_app
ENV WORKDIR ${WORKDIR}
RUN mkdir ${WORKDIR}

WORKDIR ${WORKDIR}

RUN git clone https://github.com/LBL-EESA/TECA_superbuild.git
RUN cd ${WORKDIR}/TECA_superbuild && mkdir build && cd build && \
#    cmake -DENABLE_TECA_DATA=OFF ${WORKDIR}/TECA_superbuild \
#    -DENABLE_TECA_TEST=OFF -DENABLE_OPENMPI=OFF -DENABLE_CRAY_MPICH=ON && \
#    make install
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_OPENMPI=OFF \
    -DENABLE_TECA_TEST=OFF \
    -DENABLE_TECA_DATA=OFF \
    -DENABLE_READLINE=OFF \
    ${WORKDIR}/TECA_superbuild

RUN pip3 install torch

ENV PYTHONPATH=${WORKDIR}/TECA_superbuild/build/lib:${PYTHONPATH}
ENV LD_LIBRARY_PATH=${WORKDIR}/TECA_superbuild/build/lib:${LD_LIBRARY_PATH}

CMD ["/bin/bash"]