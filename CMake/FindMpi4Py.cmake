#*****************************************************************************
# FindMpi4Py
#
# Check if mpi4py is installed
#
# This module defines
#  MPI4PY_FOUND, set TRUE if mpi4py is installed
set(_TMP_PY_OUTPUT)
set(_TMP_PY_RETURN)
exec_program("${PYTHON_EXECUTABLE}"
  ARGS "-c 'import mpi4py'"
  OUTPUT_VARIABLE _TMP_PY_OUTPUT
  RETURN_VALUE _TMP_PY_RETURN)
set(MPI4PY_FOUND FALSE)
if(NOT _TMP_PY_RETURN)
  set(MPI4PY_FOUND TRUE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPI4PY DEFAULT_MSG MPI4PY_FOUND)
