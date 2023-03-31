# - Find TELite
# Find the TELite includes and library
#
#  TELITE_INCLUDE_DIR  - user modifiable choice of where TELite headers are
#  TELITE_LIBRARY      - user modifiable choice of where TELite libraries are

#search starting from user editable cache var
if (TELITE_INCLUDE_DIR AND TELITE_LIBRARY)
  # Already in cache, be silent
  set (TELITE_FIND_QUIETLY TRUE)
endif ()

# find the library
# first look where the user told us
if (TELITE_DIR)
  find_library (TELITE_LIBRARY NAMES libTELite.a
    PATHS "${TELITE_DIR}/lib"
    NO_DEFAULT_PATH)
endif()

# next look in LD_LIBRARY_PATH for libraries
find_library (TELITE_LIBRARY NAMES libTELite.a
  PATHS ENV LD_LIBRARY_PATH NO_DEFAULT_PATH)

# finally CMake can look
find_library (TELITE_LIBRARY NAMES libTELite.a)

mark_as_advanced (TELITE_LIBRARY)

# find the header
# first look where the user told us
if (TELITE_DIR)
  find_path (TELITE_INCLUDE_DIR SimpleGrid.h
    PATHS "${TELITE_DIR}/include" NO_DEFAULT_PATH)
endif()

# then look relative to library dir
get_filename_component(TELITE_LIBRARY_DIR
  ${TELITE_LIBRARY} DIRECTORY)

find_path (TELITE_INCLUDE_DIR SimpleGrid.h
  PATHS "${TELITE_LIBRARY_DIR}/../include"
  NO_DEFAULT_PATH)

# finally CMake can look
find_path (TELITE_INCLUDE_DIR SimpleGrid.h)

mark_as_advanced (TELITE_INCLUDE_DIR)

# handle the QUIETLY and REQUIRED arguments and set TELITE_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (TELite
  DEFAULT_MSG TELITE_LIBRARY TELITE_INCLUDE_DIR)
