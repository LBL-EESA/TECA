#!/bin/bash
set -x

export DASHROOT=`pwd`
export SHA=`git log --pretty=format:'%h' -n 1`

set +x
if [[ ${DOCKER_IMAGE} == "fedora" ]]
then
    source /usr/share/Modules/init/bash
    module load mpi
fi
source `pwd`/../tci/bin/activate
set -x

export NETCDF_BUILD_TYPE="netcdf"
if [[ ${REQUIRE_NETCDF_MPI} == TRUE ]]
then
    export NETCDF_BUILD_TYPE="netcdf_mpi"
fi

export PATH=.:${PATH}
export PYTHONPATH=${DASHROOT}/build/lib
export LD_LIBRARY_PATH=${DASHROOT}/build/lib
export MPLBACKEND=Agg
mkdir build

ctest -S ${DASHROOT}/test/travis_ci/ctest_linux.cmake -VV --timeout 180
