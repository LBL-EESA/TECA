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

# find the library
# first look where the user told us
if (LIBXLSXWRITER_DIR)
  find_library (LIBXLSXWRITER_LIBRARY NAMES xlsxwriter
    PATHS "${LIBXLSXWRITER_DIR}/lib" "${LIBXLSXWRITER_DIR}/lib64"
    NO_DEFAULT_PATH)
  find_library (ZLIB_LIBRARY NAMES z zlib
    PATHS "${LIBXLSXWRITER_DIR}/lib" "${LIBXLSXWRITER_DIR}/lib64"
    NO_DEFAULT_PATH)
endif()

# next look in LD_LIBRARY_PATH for libraries
find_library (LIBXLSXWRITER_LIBRARY NAMES xlsxwriter
  PATHS ENV LD_LIBRARY_PATH NO_DEFAULT_PATH)
find_library (ZLIB_LIBRARY NAMES z zlib
  PATHS ENV LD_LIBRARY_PATH NO_DEFAULT_PATH)

# finally CMake can look
find_library (LIBXLSXWRITER_LIBRARY NAMES xlsxwriter)

mark_as_advanced (LIBXLSXWRITER_LIBRARY)

# find the header
# first look where the user told us
if (LIBXLSXWRITER_DIR)
  find_path (LIBXLSXWRITER_INCLUDE_DIR xlsxwriter.h
    PATHS "${LIBXLSXWRITER_DIR}/include" NO_DEFAULT_PATH)
  find_path (ZLIB_INCLUDE_DIR zlib.h
    PATHS "${LIBXLSXWRITER_DIR}/include" NO_DEFAULT_PATH)
endif()

# then look relative to library dir
get_filename_component(LIBXLSXWRITER_LIBRARY_DIR
  ${LIBXLSXWRITER_LIBRARY} DIRECTORY)

find_path (LIBXLSXWRITER_INCLUDE_DIR xlsxwriter.h
  PATHS "${LIBXLSXWRITER_LIBRARY_DIR}/../include"
  NO_DEFAULT_PATH)

# finally CMake can look
find_path (LIBXLSXWRITER_INCLUDE_DIR xlsxwriter.h)

mark_as_advanced (LIBXLSXWRITER_INCLUDE_DIR)

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
