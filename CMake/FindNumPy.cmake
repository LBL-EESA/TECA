#*****************************************************************************
# FindNumpy
#
# Check if numpy is installed and configure c-api includes
#
# This module defines
#  NumPy_FOUND, set TRUE if numpy and c-api are available
#  NumPy_INCLUDE_DIR, where to find c-api headers
#  NumPy_VERSION, numpy release version

set(_TMP_PY_OUTPUT)
set(_TMP_PY_RETURN)
exec_program("${PYTHON_EXECUTABLE}"
  ARGS "-c 'import numpy; print(numpy.get_include())'"
  OUTPUT_VARIABLE _TMP_PY_OUTPUT
  RETURN_VALUE _TMP_PY_RETURN)
set(NumPy_INCLUDE_FOUND FALSE)
if(NOT _TMP_PY_RETURN AND EXISTS "${_TMP_PY_OUTPUT}")
  set(NumPy_INCLUDE_FOUND TRUE)
else()
  set(_TMP_PY_OUTPUT)
endif()
set(NumPy_INCLUDE_DIR "${_TMP_PY_OUTPUT}")

set(_TMP_PY_OUTPUT)
set(_TMP_PY_RETURN)
exec_program("${PYTHON_EXECUTABLE}"
  ARGS "-c 'import numpy; print(numpy.version.version)'"
  OUTPUT_VARIABLE _TMP_PY_OUTPUT
  RETURN_VALUE _TMP_PY_RETURN)
set(NumPy_VERSION_FOUND FALSE)
if(NOT _TMP_PY_RETURN)
  set(NumPy_VERSION_FOUND TRUE)
  message(STATUS "Looking for Python package numpy ... found version ${_TMP_PY_OUTPUT}")
else()
  set(_TMP_PY_OUTPUT)
  message(STATUS "Looking for Python package numpy ... not found")
endif()
set(NumPy_VERSION "${_TMP_PY_OUTPUT}")

#set(NumPy_INCLUDE_DIR "${_TMP_PY_OUTPUT}" CACHE PATH "Numpy C API headers")
#mark_as_advanced(NumPy_INCLUDE_DIR)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy DEFAULT_MSG NumPy_INCLUDE_FOUND NumPy_VERSION_FOUND)
