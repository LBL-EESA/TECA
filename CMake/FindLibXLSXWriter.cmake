# - Find LibXLSWriter
# Find the LibXLSWriter includes and library
#
#  LIBXLSXWRITER_INCLUDE_DIR  - user modifiable choice of where netcdf headers are
#  LIBXLSXWRITER_LIBRARY      - user modifiable choice of where netcdf libraries are

#search starting from user editable cache var
if(LIBXLSXWRITER_INCLUDE_DIR AND LIBXLSXWRITER_LIBRARY)
  # Already in cache, be silent
  set(LIBXLSXWRITER_FIND_QUIETLY TRUE)
endif()

set(USE_DEFAULT_PATHS "NO_DEFAULT_PATH")
if(LIBXLSXWRITER_USE_DEFAULT_PATHS)
  set(USE_DEFAULT_PATHS "")
endif()

find_path(LIBXLSXWRITER_INCLUDE_DIR xlsxwriter.h
  HINTS "${LIBXLSXWRITER_DIR}/include")
mark_as_advanced(LIBXLSXWRITER_INCLUDE_DIR)

find_library(LIBXLSXWRITER_LIBRARY NAMES xlsxwriter
  HINTS "${LIBXLSXWRITER_DIR}/lib")
mark_as_advanced(LIBXLSXWRITER_LIBRARY)

find_package(ZLIB REQUIRED)

set(LIBXLSXWRITER_LIBRARIES
    ${LIBXLSXWRITER_LIBRARY} ${ZLIB_LIBRARIES} CACHE STRING
    "link time dpendencies of the libxlswriter library")
mark_as_advanced(LIBXLSXWRITER_LIBRARIES)

# handle the QUIETLY and REQUIRED arguments and set LIBXLSXWRITER_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (LibXLSXWriter
  DEFAULT_MSG LIBXLSXWRITER_LIBRARY LIBXLSXWRITER_INCLUDE_DIR)
