#!/bin/bash
set -x

export DASHROOT=`pwd`
export SHA=`git log --pretty=format:'%h' -n 1`
export PATH=.:/usr/local/bin:${PATH}
export PYTHONPATH=${TRAVIS_BUILD_DIR}/build/lib
export LD_LIBRARY_PATH=${TRAVIS_BUILD_DIR}/build/lib
export DYLD_LIBRARY_PATH=${TRAVIS_BUILD_DIR}/build/lib
export MPLBACKEND=Agg
export TMPDIR=/tmp

set +x
source `pwd`/../tci/bin/activate
set -x

export PATH=$(brew --prefix)/opt/curl/bin:$PATH
export DYLD_LIBRARY_PATH=$(brew --prefix)/opt/curl/lib:$DYLD_LIBRARY_PATH

mkdir build
ctest -S ${DASHROOT}/test/travis_ci/ctest_osx.cmake --output-on-failure --timeout 400 &
ctest_pid=$!

# this loop prevents travis from killing the job
set +x
while [ -n "$(ps -p $ctest_pid -o pid=)" ]
do
  echo "ctest pid=$ctest_pid alive"
  sleep 30s
done

# return the exit code from ctest
wait $ctest_pid
exit $?
