#*****************************************************************************
# FindTcpyPI
#
# Check if tcpyPI is installed
#
# This module defines
#  TcpyPI_FOUND, set TRUE if tcpyPI package is available

set(_TMP_PY_OUTPUT)
set(_TMP_PY_RETURN)
exec_program("${PYTHON_EXECUTABLE}"
  ARGS "-c 'import tcpyPI'"
  OUTPUT_VARIABLE _TMP_PY_OUTPUT
  RETURN_VALUE _TMP_PY_RETURN)

set(TcpyPI_PACKAGE_FOUND FALSE)
if(NOT _TMP_PY_RETURN)
  set(TcpyPI_PACKAGE_FOUND TRUE)
  message(STATUS "Looking for tcpyPI package tcpyPI ... found")
else()
  message(STATUS "Looking for tcpyPI package tcpyPI ... not found")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TcpyPI DEFAULT_MSG TcpyPI_PACKAGE_FOUND)
