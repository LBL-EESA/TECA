#!/bin/bash
set -v
export DASHROOT=`pwd`
#export SHA=`git --git-dir=${TECA_DIR}/.git log --pretty=format:'%h' -n 1`
export SHA=`git log --pretty=format:'%h' -n 1`
if [[ "$DOCKER_IMAGE" == "fedora" ]]; then
    source /usr/share/Modules/init/bash
    module load mpi
fi
export PATH=.:${PATH}
export PYTHONPATH=${DASHROOT}/build/lib.linux-x86_64-2.7/teca/lib
export LD_LIBRARY_PATH=${DASHROOT}/build/lib.linux-x86_64-2.7/teca/lib
export MPLBACKEND=Agg
mkdir build
cmake --version

python${TECA_PYTHON_VERSION} setup.py install
