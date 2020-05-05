FROM ubuntu:18.04

ARG USER=teca
ENV USER ${USER}

# HOME must be added for NERSC to be able to load 
# the installed packages from ~/.local
ENV DEBIAN_FRONTEND=noninteractive \
    USER_HOME=/home/${USER} \ 
    HOME=/home/${USER}


RUN apt-get update -y && \
    apt-get install -y git-core gcc g++ gfortran cmake swig \
        libmpich-dev libhdf5-dev libnetcdf-dev libboost-program-options-dev \
        python3-dev python3-pip subversion libudunits2-0 \
        libudunits2-dev zlib1g-dev libssl-dev && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#### ADD DEFAULT USER ####
RUN adduser --disabled-password --gecos "" ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN chown -R ${USER}:${USER} ${USER_HOME}

#### CREATE WORKING DIRECTORY FOR USER ####
ARG WORKDIR=/teca_app
ENV WORKDIR ${WORKDIR}
RUN mkdir ${WORKDIR}
RUN chown -R ${USER}:${USER} ${WORKDIR}

WORKDIR ${WORKDIR}

RUN pip3 install --upgrade pip

USER ${USER}

RUN pip3 install --user numpy mpi4py matplotlib torch

RUN svn co svn://svn.code.sf.net/p/teca/TECA_assets TECA_assets 

RUN git clone --single-branch --branch teca_assets_svn_support https://github.com/LBL-EESA/TECA.git
RUN cd ${WORKDIR}/TECA && mkdir build && cd build && \
    cmake -DTECA_ASSETS_ROOT=${WORKDIR}/TECA_assets/ ${WORKDIR}/TECA

USER root

RUN cd ${WORKDIR}/TECA/build && make install

USER ${USER}

ENV PYTHONPATH=${WORKDIR}/TECA/build/lib:${PYTHONPATH}
ENV LD_LIBRARY_PATH=${WORKDIR}/TECA/build/lib:${LD_LIBRARY_PATH}

CMD ["/bin/bash"]