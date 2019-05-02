FROM ubuntu:18.04

RUN apt-get update -y && \
	apt-get install -y git-core g++ cmake pkg-config \
	subversion libexpat-dev zlib1g-dev libpcre3-dev gfortran

#### CREATE WORKING DIRECTORY FOR USER ####
ARG WORKDIR=/project
ENV WORKDIR ${WORKDIR}
RUN mkdir -p ${WORKDIR}/TECA_build

WORKDIR ${WORKDIR}

ARG TECA_INSTALL_PREFIX=/teca_deps/2.1.3
ENV TECA_INSTALL_PREFIX ${TECA_INSTALL_PREFIX}
RUN mkdir -p ${TECA_INSTALL_PREFIX}

#### ADD DEFAULT USER ####
ARG USER=cascade
ENV USER ${USER}
RUN adduser --disabled-password --gecos "" ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN chown -R ${USER}:${USER} ${WORKDIR}
RUN chown -R ${USER}:${USER} ${TECA_INSTALL_PREFIX}

USER ${USER}

#### SETTING ENV VARIABLES ####
ENV WORKDIR ${WORKDIR}
ENV TECA_INSTALL_PREFIX ${TECA_INSTALL_PREFIX}
#ENV PYTHONPATH "${WORKDIR}/TECA_build/lib:${PYTHONPATH}"
#ENV LD_LIBRARY_PATH "${WORKDIR}/TECA_build/lib:${LD_LIBRARY_PATH}"
ENV MPLBACKEND "Agg"

RUN git clone https://github.com/LBL-EESA/TECA_superbuild.git
RUN cd TECA_superbuild && mkdir build && cd build && \
	cmake -DCMAKE_INSTALL_PREFIX=${TECA_INSTALL_PREFIX} -DENABLE_TECA=OFF .. \
	&& make -j8 && make install
RUN /bin/bash -c "source ${TECA_INSTALL_PREFIX}/bin/teca_env.sh"

CMD ["/bin/bash"]