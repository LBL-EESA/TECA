#*****************************************************************************
# FindPyTorch
#
# Check if torch is installed and configure c-api includes
#
# This module defines
#  PyTorch_FOUND, set TRUE if torch and c-api are available
#  PyTorch_VERSION, torch release version

set(_TMP_PY_OUTPUT)
set(_TMP_PY_RETURN)
exec_program("${PYTHON_EXECUTABLE}"
  ARGS "-c 'import torch; print(torch.__version__)'"
  OUTPUT_VARIABLE _TMP_PY_OUTPUT
  RETURN_VALUE _TMP_PY_RETURN)
set(PyTorch_VERSION_FOUND FALSE)
if(NOT _TMP_PY_RETURN)
  set(PyTorch_VERSION_FOUND TRUE)
  message(STATUS "Looking for Python package pytorch ... found version ${_TMP_PY_OUTPUT}")
else()
  set(_TMP_PY_OUTPUT)
  message(STATUS "Looking for Python package pytorch ... not found")
endif()
set(PyTorch_VERSION "${_TMP_PY_OUTPUT}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PyTorch DEFAULT_MSG PyTorch_VERSION_FOUND)
