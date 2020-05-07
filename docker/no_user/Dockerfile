FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y git-core gcc g++ gfortran cmake swig \
        libhdf5-dev libnetcdf-dev libboost-program-options-dev \
        python3-dev python3-pip subversion libudunits2-0 \
        libudunits2-dev zlib1g-dev libssl-dev wget && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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

ARG mpi4py=3.0.0
ARG mpi4py_prefix=mpi4py-$mpi4py

RUN \
    wget https://bitbucket.org/mpi4py/mpi4py/downloads/$mpi4py_prefix.tar.gz && \
    tar xvzf $mpi4py_prefix.tar.gz                                           && \
    cd $mpi4py_prefix                                                        && \
    python3 setup.py build                                                   && \
    python3 setup.py install                                                 && \
    cd ..                                                                    && \
    rm -rf $mpi4py_prefix

RUN /sbin/ldconfig

#### CREATE WORKING DIRECTORY FOR USER ####
ARG WORKDIR=/teca_app
ENV WORKDIR ${WORKDIR}
RUN mkdir ${WORKDIR}

WORKDIR ${WORKDIR}

RUN pip3 install --upgrade pip

RUN pip3 install numpy mpi4py matplotlib torch

RUN svn co svn://svn.code.sf.net/p/teca/TECA_assets TECA_assets 

RUN git clone --single-branch --branch teca_assets_svn_support https://github.com/LBL-EESA/TECA.git
RUN cd ${WORKDIR}/TECA && mkdir build && cd build && \
    cmake -DTECA_ASSETS_ROOT=${WORKDIR}/TECA_assets/ ${WORKDIR}/TECA && make install

ENV PYTHONPATH=${WORKDIR}/TECA/build/lib:${PYTHONPATH}
ENV LD_LIBRARY_PATH=${WORKDIR}/TECA/build/lib:${LD_LIBRARY_PATH}

CMD ["/bin/bash"]