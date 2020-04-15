# - Find NetCDF
# Find the native NetCDF includes and library
#
#  NETCDF_INCLUDE_DIR  - user modifiable choice of where netcdf headers are
#  NETCDF_LIBRARY      - user modifiable choice of where netcdf libraries are
#
# Your package can require certain interfaces to be FOUND by setting these
#
#  NETCDF_CXX         - require the C++ interface and link the C++ library
#  NETCDF_F77         - require the F77 interface and link the fortran library
#  NETCDF_F90         - require the F90 interface and link the fortran library
#
# Or equivalently by calling FindNetCDF with a COMPONENTS argument containing one or
# more of "CXX;F77;F90".
#
# When interfaces are requested the user has access to interface specific hints:
#
#  NETCDF_${LANG}_INCLUDE_DIR - where to search for interface header files
#  NETCDF_${LANG}_LIBRARY     - where to search for interface libraries
#
# This module returns these variables for the rest of the project to use.
#
#  NETCDF_FOUND          - True if NetCDF found including required interfaces (see below)
#  NETCDF_LIBRARIES      - All netcdf related libraries.
#  NETCDF_INCLUDE_DIRS   - All directories to include.
#  NETCDF_HAS_INTERFACES - Whether requested interfaces were found or not.
#  NETCDF_${LANG}_INCLUDE_DIRS/NETCDF_${LANG}_LIBRARIES - C/C++/F70/F90 only interface
#
# Normal usage would be:
#  set (NETCDF_F90 "YES")
#  find_package (NetCDF REQUIRED)
#  target_link_libraries (uses_everthing ${NETCDF_LIBRARIES})
#  target_link_libraries (only_uses_f90 ${NETCDF_F90_LIBRARIES})


#search starting from user editable cache var
if (NETCDF_INCLUDE_DIR AND NETCDF_LIBRARY)
  # Already in cache, be silent
  set (NETCDF_FIND_QUIETLY TRUE)
endif()

# find the library
# use package config, this works well on systems that make
# use of modules to manage installs not in the standard
# locations that cmake knows about
find_package(PkgConfig REQUIRED)
pkg_check_modules(NC_TMP netcdf QUIET)
if (NC_TMP_FOUND AND NC_TMP_LINK_LIBRARIES AND NC_TMP_LIBRARY_DIRS AND NC_TMP_INCLUDE_DIRS)
    set(NETCDF_LIBRARY_DIR ${NC_TMP_LIBRARY_DIRS})
    set(NETCDF_LIBRARY ${NC_TMP_LINK_LIBRARIES})
    set(NETCDF_INCLUDE_DIR ${NC_TMP_INCLUDE_DIRS})
else()
    # package config failed, use cmake
    # first look where the user told us
    if (NETCDF_DIR)
      find_library(NETCDF_LIBRARY NAMES netcdf
        PATHS "${NETCDF_DIR}/lib" "${NETCDF_DIR}/lib64"
        NO_DEFAULT_PATH)
    endif()

    # next look in LD_LIBRARY_PATH for libraries
    find_library(NETCDF_LIBRARY NAMES netcdf
      PATHS ENV LD_LIBRARY_PATH NO_DEFAULT_PATH)

    # finally CMake can look
    find_library(NETCDF_LIBRARY NAMES netcdf)

    message(STATUS ${NETCDF_LIBRARY})
endif()

# if we can find the library it is found now
# record what we have
mark_as_advanced (NETCDF_LIBRARY)
set (NETCDF_C_LIBRARIES ${NETCDF_LIBRARY})

# find the header
# package config failed, use cmake
if (NOT NC_TMP_FOUND OR NOT NC_TMP_LINK_LIBRARIES OR NOT NC_TMP_LIBRARY_DIRS OR NOT NC_TMP_INCLUDE_DIRS)
    # first look where the user told us
    if (NETCDF_DIR)
      find_path (NETCDF_INCLUDE_DIR netcdf.h
        PATHS "${NETCDF_DIR}/include" NO_DEFAULT_PATH)
    endif()

    # then look relative to library dir
    get_filename_component(NETCDF_LIBRARY_DIR
      ${NETCDF_LIBRARY} DIRECTORY)

    find_path (NETCDF_INCLUDE_DIR netcdf.h
      PATHS "${NETCDF_LIBRARY_DIR}/../include"
      NO_DEFAULT_PATH)

    # CMake can look
    find_path(NETCDF_INCLUDE_DIR netcdf.h)
endif()

# look for header file that indicates MPI support
set(NETCDF_IS_PARALLEL FALSE)
find_file(NETCDF_PAR_INCLUDE_DIR netcdf_par.h
    PATHS ${NETCDF_INCLUDE_DIR} NO_DEFAULT_PATH)
if (NETCDF_PAR_INCLUDE_DIR)
    set(NETCDF_IS_PARALLEL TRUE)
endif()

# if we can find the headers they are found now
# record what we have
mark_as_advanced(NETCDF_INCLUDE_DIR)
mark_as_advanced(NETCDF_IS_PARALLEL)
mark_as_advanced(NETCDF_PAR_INCLUDE_DIR)
set(NETCDF_C_INCLUDE_DIRS ${NETCDF_INCLUDE_DIR})

#start finding requested language components
set (NetCDF_libs "")
set (NetCDF_includes "${NETCDF_INCLUDE_DIR}")

get_filename_component (NetCDF_lib_dirs "${NETCDF_LIBRARY}" PATH)

#export accumulated results to internal varS that rest of project can depend on
list(APPEND NetCDF_libs "${NETCDF_C_LIBRARIES}")
set(NETCDF_LIBRARIES ${NetCDF_libs})
set(NETCDF_INCLUDE_DIRS ${NetCDF_includes})

# handle the QUIETLY and REQUIRED arguments and set NETCDF_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (NetCDF
  DEFAULT_MSG NETCDF_LIBRARIES NETCDF_INCLUDE_DIRS)
