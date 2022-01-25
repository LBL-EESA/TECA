#*****************************************************************************
# FindCupy
#
# Check if cupy is installed and configure c-api includes
#
# This module defines
#  CuPy_FOUND, set TRUE if cupy and c-api are available
#  CuPy_VERSION, cupy release version

set(_TMP_PY_OUTPUT)
set(_TMP_PY_RETURN)
exec_program("${PYTHON_EXECUTABLE}"
  ARGS "-c 'import cupy; print(cupy.__version__)'"
  OUTPUT_VARIABLE _TMP_PY_OUTPUT
  RETURN_VALUE _TMP_PY_RETURN)
set(CuPy_VERSION_FOUND FALSE)
if(NOT _TMP_PY_RETURN)
  set(CuPy_VERSION_FOUND TRUE)
  message(STATUS "Looking for Python package cupy ... found version ${_TMP_PY_OUTPUT}")
else()
  set(_TMP_PY_OUTPUT)
  message(STATUS "Looking for Python package cupy ... not found")
endif()
set(CuPy_VERSION "${_TMP_PY_OUTPUT}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CuPy DEFAULT_MSG CuPy_VERSION_FOUND)
