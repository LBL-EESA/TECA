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
ENV PYTHONPATH "${WORKDIR}/TECA/build/lib:${PYTHONPATH}"
ENV LD_LIBRARY_PATH "${WORKDIR}/TECA/build/lib:${LD_LIBRARY_PATH}"
ENV MPLBACKEND "Agg"

RUN git clone https://github.com/LBL-EESA/TECA_superbuild.git
RUN cd TECA_superbuild && mkdir build && cd build && \
	cmake -DCMAKE_INSTALL_PREFIX=${TECA_INSTALL_PREFIX} -DENABLE_TECA=OFF .. \
	&& make -j8 && make install
RUN /bin/bash -c "source ${TECA_INSTALL_PREFIX}/bin/teca_env.sh"

#RUN svn co svn://missmarple.lbl.gov/work3/teca/TECA_data TECA_data

#RUN cd TECA && mkdir build && cd build && \
#	cmake -DBUILD_TESTING=ON -DTECA_DATA_ROOT=${WORKDIR}/TECA_data/ \
#	-DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/teca-deps/2.1.3/ \
#	-DBOOST_ROOT=/teca_deps/2.1.3/include/ -DNETCDF_DIR=/teca_deps/2.1.3/ \
#	-DTECA_PYTHON_VERSION=2 .

#### ADD DEFAULT USER ####
#ARG USER=cascade
#ENV USER ${USER}
#RUN adduser --disabled-password --gecos "" ${USER} && \
#    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

#RUN chown -R ${USER}:${USER} ${WORKDIR}

#USER ${USER}

CMD ["/bin/bash"]