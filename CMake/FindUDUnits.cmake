# - Find UDUnits
# Find the UDUnits includes and library
#
#  UDUNITS_INCLUDE_DIR  - user modifiable choice of where netcdf headers are
#  UDUNITS_LIBRARY      - user modifiable choice of where netcdf libraries are

#search starting from user editable cache var
if (UDUNITS_INCLUDE_DIR AND UDUNITS_LIBRARY)
  # Already in cache, be silent
  set (UDUNITS_FIND_QUIETLY TRUE)
endif ()

set(USE_DEFAULT_PATHS "NO_DEFAULT_PATH")
if(UDUNITS_USE_DEFAULT_PATHS)
  set(USE_DEFAULT_PATHS "")
endif()

find_path (UDUNITS_INCLUDE_DIR udunits2.h
  HINTS "${UDUNITS_DIR}/include")
mark_as_advanced (UDUNITS_INCLUDE_DIR)

find_library (UDUNITS_LIBRARY NAMES udunits2
  HINTS "${UDUNITS_DIR}/lib")
mark_as_advanced (UDUNITS_LIBRARY)

# handle the QUIETLY and REQUIRED arguments and set UDUNITS_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (UDUnits
  DEFAULT_MSG UDUNITS_LIBRARY UDUNITS_INCLUDE_DIR)
