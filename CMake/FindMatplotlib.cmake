#*****************************************************************************
# FindMatplotlib
#
# Check if matplotlib is installed and configure c-api includes
#
# This module defines
#  Matplotlib_FOUND, set TRUE if matplotlib and c-api are available
#  Matplotlib_VERSION, matplotlib release version

set(_TMP_PY_OUTPUT)
set(_TMP_PY_RETURN)
exec_program("${PYTHON_EXECUTABLE}"
  ARGS "-c 'import matplotlib; print(matplotlib.__version__)'"
  OUTPUT_VARIABLE _TMP_PY_OUTPUT
  RETURN_VALUE _TMP_PY_RETURN)
set(Matplotlib_VERSION_FOUND FALSE)
if(NOT _TMP_PY_RETURN)
  set(Matplotlib_VERSION_FOUND TRUE)
else()
  set(_TMP_PY_OUTPUT)
endif()
set(Matplotlib_VERSION "${_TMP_PY_OUTPUT}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Matplotlib DEFAULT_MSG Matplotlib_VERSION_FOUND)
