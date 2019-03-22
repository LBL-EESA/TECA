cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0028 NEW)
cmake_policy(SET CMP0053 NEW)

project(teca C CXX Fortran)

# set build/install sub dirs for various components
if (NOT LIB_PREFIX)
  set(LIB_PREFIX lib)
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LIB_PREFIX})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LIB_PREFIX})
if (NOT BIN_PREFIX)
  set(BIN_PREFIX bin)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${BIN_PREFIX})
if (NOT INCLUDE_PREFIX)
  set(INCLUDE_PREFIX include)
endif()

# bring our cmake codes into scope
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# if we forgot to set the build type default to release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release"
      CACHE STRING "One of: Debug Release RelWithDebInfo MinSizeRel"
      FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
        STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Configuring a ${CMAKE_BUILD_TYPE} build")

# add the requisite flags to enable c++11 features
# on various platforms.
if (MSVC)
    # TODO require vs2015
    # enable exceptions
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc"
            CACHE STRING "" FORCE)
else()
    if (NOT CMAKE_CXX_FLAGS)
        set(tmp "-fPIC -std=c++11 -Wall -Wextra")
        if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
            set(tmp "${tmp} -stdlib=libc++")
        endif()
        if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
            set(tmp "${tmp} -march=native -mtune=native")
        endif()
        set(CMAKE_CXX_FLAGS "${tmp}"
            CACHE STRING "TECA build defaults"
            FORCE)
    endif()
    if (NOT CMAKE_Fortran_FLAGS)
        set(tmp "-Wall -Wextra -Wno-conversion")
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            set(tmp "${tmp} -march=native -mtune=native")
        endif()
        set(CMAKE_Fortran_FLAGS "${tmp}"
            CACHE STRING "TECA build defaults"
            FORCE)
    endif()
endif()

# platform specific compile time feature detection
message(STATUS "Check for c++ regex support")
try_run(CXX11_REGEX_WORKS CXX11_REGEX_COMPILES
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/CMake/teca_platform_tests.cpp"
    CMAKE_FLAGS ${CMAKE_CXX_FLAGS}
    COMPILE_DEFINITIONS "-DCXX11_REGEX_TEST")
if (NOT (CXX11_REGEX_COMPILES AND CXX11_REGEX_WORKS))
    message(STATUS "Check for c++ regex support -- not found. Use a c++11 compiler to enable.")
else()
    message(STATUS "Check for c++ regex support -- enabled")
    set(TECA_HAS_REGEX ON CACHE BOOL "Regex features")
endif()

# locate dependencies
# configure for MPI
if (ENABLE_CRAY_MPICH)
    set(ENV{PKG_CONFIG_PATH} "$ENV{CRAY_MPICH_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    find_package(PkgConfig QUIET)
    pkg_check_modules(CRAY_MPICH REQUIRED QUIET mpich)
    set(MPI_C_INCLUDE_PATH ${CRAY_MPICH_INCLUDE_DIRS} CACHE STRING "MPI include directories")
    set(MPI_C_LIBRARIES ${CRAY_MPICH_LDFLAGS} CACHE STRING "MPI link dependencies")
    set(MPIEXEC srun CACHE STRING "Platform MPI run equivalent")
    set(MPI_C_FOUND CACHE BOOL ON "status of MPI config")
else()
    find_package(MPI)
endif()
set(tmp OFF)
if (MPI_C_FOUND  AND ((DEFINED TECA_HAS_MPI AND TECA_HAS_MPI)
    OR (NOT DEFINED TECA_HAS_MPI)))
    message(STATUS "MPI features -- enabled")
    set(tmp ON)
elseif (REQUIRE_MPI)
    message(FATAL_ERROR "MPI features -- required but not found.")
else()
    message(STATUS "MPI features -- not found.")
endif()
set(TECA_HAS_MPI ${tmp} CACHE BOOL "MPI features")

# configure for NetCDF
set(tmp OFF)
find_package(NetCDF)
if (NETCDF_FOUND AND ((DEFINED TECA_HAS_NETCDF AND TECA_HAS_NETCDF)
    OR (NOT DEFINED TECA_HAS_NETCDF)))
    message(STATUS "NetCDF features -- enabled")
    set(tmp ON)
elseif (REQUIRE_NETCDF)
    message(FATAL_ERROR "NetCDF features -- required but not found. set NETCDF_DIR to enable.")
else()
    message(STATUS "NetCDF features -- not found. set NETCDF_DIR to enable.")
    message(WARNING "NetCDF is required for CF-2 I/O")
endif()
set(TECA_HAS_NETCDF ${tmp} CACHE BOOL "NetCDF features")

# configure for libxlsxwriter
set(tmp OFF)
find_package(LibXLSXWriter)
if (LIBXLSXWRITER_FOUND AND ((DEFINED TECA_HAS_LIBXLSXWRITER AND TECA_HAS_LIBXLSXWRITER)
    OR (NOT DEFINED TECA_HAS_LIBXLSXWRITER)))
    message(STATUS "libxlsxwriter features -- enabled")
    set(tmp ON)
elseif (REQUIRE_LIBXLSXWRITE)
    message(STATUS "libxlsxwriter features -- required but not found. set LIBXLSXWRITER_DIR to enable.")
else()
    message(STATUS "libxlsxwriter features -- not found. set LIBXLSXWRITER_DIR to enable.")
endif()
set(TECA_HAS_LIBXLSXWRITER ${tmp} CACHE BOOL "libxlsxwriter features")

# configure for UDUnits
set(tmp OFF)
find_package(UDUnits)
if (UDUNITS_FOUND AND ((DEFINED TECA_HAS_UDUNITS AND TECA_HAS_UDUNITS)
    OR (NOT DEFINED TECA_HAS_UDUNITS)))
    message(STATUS "UDUnits features -- enabled")
    set(tmp ON)
elseif (REQUIRE_UDUNITS)
    message(FATAL_ERROR "UDUnits features -- required but not found. set UDUNITS_DIR to enable.")
else()
    message(STATUS "UDUnits features -- not found. set UDUNITS_DIR to enable.")
endif()
set(TECA_HAS_UDUNITS ${tmp} CACHE BOOL "UDUnits features")

# configure for ParaView
set(tmp OFF)
find_package(ParaView QUIET)
if (ParaView_FOUND AND ((DEFINED TECA_HAS_PARAVIEW AND TECA_HAS_PARAVIEW) OR (NOT DEFINED TECA_HAS_PARAVIEW)))
    message(STATUS "ParaView features -- enabled")
    set(tmp ON)
elseif (REQUIRE_PARAVIEW)
    message(FATAL_ERROR "ParaView features -- required but not found. set ParaView_DIR to enable.")
else()
    message(STATUS "ParaView features -- not found. set ParaView_DIR to enable.")
endif()
set(TECA_HAS_PARAVIEW ${tmp} CACHE BOOL "ParaView features")

# configure for VTK
set(tmp OFF)
if (NOT TECA_HAS_PARAVIEW)
    find_package(VTK QUIET)
    if (VTK_FOUND AND ((DEFINED TECA_HAS_VTK AND TECA_HAS_VTK) OR (NOT DEFINED TECA_HAS_VTK)))
        message(STATUS "VTK features -- enabled")
        set(tmp ON)
    elseif (REQUIRE_VTK)
        message(FATAL_ERROR "VTK features -- required but not found. set VTK_DIR to enable.")
    else()
        message(STATUS "VTK features -- not found. set VTK_DIR to enable.")
    endif()
endif()
set(TECA_HAS_VTK ${tmp} CACHE BOOL "VTK features")

#configure for Boost
set(tmp OFF)
find_package(Boost QUIET COMPONENTS program_options)
if (Boost_FOUND AND ((DEFINED TECA_HAS_BOOST AND TECA_HAS_BOOST) OR (NOT DEFINED TECA_HAS_BOOST)))
    message(STATUS "Boost features -- enabled")
    set(tmp ON)
elseif (REQUIRE_BOOST)
    message(FATAL_ERROR "Boost fetaures -- required but not found. set BOOST_ROOT to enable.")
else()
    message(STATUS "Boost fetaures -- not found. set BOOST_ROOT to enable.")
    message(WARNING "Boost is required for command line applications")
endif()
set(TECA_HAS_BOOST ${tmp} CACHE BOOL "Boost features")

#configure for OpenSSL
set(tmp OFF)
find_package(OpenSSL QUIET)
if (OpenSSL_FOUND AND ((DEFINED TECA_HAS_OPENSSL AND TECA_HAS_OPENSSL) OR (NOT DEFINED TECA_HAS_OPENSSL)))
    message(STATUS "OpenSSL features -- enabled")
    set(tmp ON)
elseif (REQUIRE_OPENSSL)
    message(FATAL_ERROR "OpenSSL fetaures -- required but not found. set OPENSSL_ROOT_DIR to enable.")
else()
    message(STATUS "OpenSSL fetaures -- not found. set OPENSSL_ROOT_DIR to enable.")
endif()
set(TECA_HAS_OPENSSL ${tmp} CACHE BOOL "OpenSSL features")

# configure for Python
set(TECA_PYTHON_VERSION 3 CACHE STRING "The major version number of Python TECA should use.")
set_property(CACHE TECA_PYTHON_VERSION PROPERTY STRINGS 2 3)
set(tmp OFF)
find_package(PythonInterp ${TECA_PYTHON_VERSION})
if(PYTHONINTERP_FOUND)
    if (TECA_PYTHON_VERSION EQUAL 3)
        set(py_lib_ver ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}m)
    else()
        set(py_lib_ver ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
    endif()
    find_program(PYTHON_CONFIG_EXECUTABLE python${PYTHON_VERSION_MAJOR}-config)
    if (NOT PYTHON_CONFIG_EXECUTABLE)
        message(SEND_ERROR "python${PYTHON_VERSION_MAJOR}-config executable is required.")
    endif()
    execute_process(COMMAND ${PYTHON_CONFIG_EXECUTABLE} --prefix
        OUTPUT_VARIABLE python_prefix OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(PYTHON_INCLUDE_DIR ${python_prefix}/include/python${py_lib_ver})
    if (EXISTS ${python_prefix}/lib/libpython${py_lib_ver}${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(PYTHON_LIBRARY ${python_prefix}/lib/libpython${py_lib_ver}${CMAKE_SHARED_LIBRARY_SUFFIX})
    elseif (EXISTS ${python_prefix}/lib64/libpython${py_lib_ver}${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(PYTHON_LIBRARY ${python_prefix}/lib64/libpython${py_lib_ver}${CMAKE_SHARED_LIBRARY_SUFFIX})
    elseif (EXISTS ${python_prefix}/lib/x86_64-linux-gnu/libpython${py_lib_ver}${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(PYTHON_LIBRARY ${python_prefix}/lib/x86_64-linux-gnu/libpython${py_lib_ver}${CMAKE_SHARED_LIBRARY_SUFFIX})
    else()
        message(SEND_ERROR "Failed to locate Python library for ${python_prefix}")
    endif()
endif()
find_package(PythonLibs ${TECA_PYTHON_VERSION})
find_package(Numpy)
find_program(swig_cmd NAMES swig swig3.0)
find_package(Mpi4Py)
if ((PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND NUMPY_FOUND AND swig_cmd)
    AND ((DEFINED TECA_HAS_PYTHON AND TECA_HAS_PYTHON)
    OR (NOT DEFINED TECA_HAS_PYTHON)))
    message(STATUS "Python ${TECA_PYTHON_VERSION} features -- enabled")
    set(tmp ON)
elseif (REQUIRE_PYTHON)
    message(FATAL_ERROR "Python ${TECA_PYTHON_VERSION} features -- required but not found")
else()
    message(STATUS "Python ${TECA_PYTHON_VERSION} features -- not found. Missing SWIG, numpy, and mpi4py")
endif()
set(TECA_HAS_PYTHON ${tmp} CACHE BOOL "Python binding")

# silence the annoying cmake warnings about REQUIRE_<X> when
# the required package X is actually found.
set(ignore ${REQUIRE_MPI} ${REQUIRE_NETCDF} ${REQUIRE_LIBXLSXWRITE}
    ${REQUIRE_UDUNITS} ${REQUIRE_PARAVIEW} ${REQUIRE_VTK} ${REQUIRE_BOOST}
    ${REQUIRE_OPENSSL} ${REQUIRE_PYTHON})

# configure library type
set(CMAKE_MACOSX_RPATH 1)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
if (TECA_HAS_PYTHON)
    # SWIG RTTI system requires libs when multiple modules
    # have cross dependencies
    set(BUILD_SHARED_LIBS ON FORCE)
endif()

# get the build version descriptor
if (NOT DEFINED TECA_VERSION)
    set(tmp)
    find_package(Git)
    if (GIT_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE}
            --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git describe --tags
            OUTPUT_VARIABLE tmp OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    if (NOT tmp)
        set(tmp "0.0.0")
    endif()
else()
    set(tmp ${TECA_VERSION})
endif()
set(TECA_VERSION_DESCR ${tmp} CACHE STRING "TECA version descriptor")
message(STATUS "TECA version ${TECA_VERSION_DESCR}")

# generate global configuration
include(CMake/teca_test.cmake)
include(CMake/teca_app.cmake)
include(CMake/teca_python.cmake)
configure_file(teca_config.h.in teca_config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# build library components and applications
add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(io)
add_subdirectory(alg)
add_subdirectory(python)
add_subdirectory(system)
add_subdirectory(apps)
add_subdirectory(paraview)

# enable regression tests
set(BUILD_TESTING OFF CACHE BOOL "Enable tests")
if (BUILD_TESTING)
    include(CTest)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake"
        @ONLY)
    set(TECA_DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../TECA_data"
        CACHE PATH "Path to TECA test data directory")
    if (TECA_DATA_ROOT AND (NOT EXISTS "${TECA_DATA_ROOT}"))
        message(STATUS "Test data -- not available")
    else()
        message(STATUS "Test data -- available")
    endif()
    set(TECA_TEST_CORES 4 CACHE STRING
        "Number of cores for use in parallel tests")
    math(EXPR HALF_CORES "${TECA_TEST_CORES}/2")
    if (HALF_CORES LESS 1)
        message(FATAL_ERROR "Parallel test require at lest 2 cores")
    endif()
    add_subdirectory(test)
endif()
