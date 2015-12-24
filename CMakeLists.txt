cmake_minimum_required(VERSION 2.8.12)

project(teca)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# add the requisite flags to enable c++11 features
# on various platforms
if (MSVC)
    # TODO require vs2013
    # enable exceptions
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -Wall -Wextra")
    if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
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
set(tmp OFF)
find_package(MPI)
if (MPI_C_FOUND  AND ((DEFINED TECA_HAS_MPI AND TECA_HAS_MPI)
    OR (NOT DEFINED TECA_HAS_MPI)))
    message(STATUS "MPI features -- enabled")
    set(tmp ON)
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
else()
    message(STATUS "NetCDF features -- not found. set NETCDF_DIR to enable.")
    message(WARNING "NetCDF is required for CF-2 I/O")
endif()
set(TECA_HAS_NETCDF ${tmp} CACHE BOOL "NetCDF features")

# configure for VTK
set(tmp OFF)
find_package(VTK QUIET)
if (VTK_FOUND AND ((DEFINED TECA_HAS_VTK AND TECA_HAS_VTK) OR (NOT DEFINED TECA_HAS_VTK)))
    message(STATUS "VTK features -- enabled")
    set(tmp ON)
else()
    message(STATUS "VTK features -- not found. set VTK_DIR to enable.")
endif()
set(TECA_HAS_VTK ${tmp} CACHE BOOL "VTK features")

#configure for Boost
set(tmp OFF)
find_package(Boost QUIET COMPONENTS program_options)
if (Boost_FOUND AND ((DEFINED TECA_HAS_BOOST AND TECA_HAS_BOOST) OR (NOT DEFINED TECA_HAS_BOOST)))
    message(STATUS "Boost features -- enabled")
    set(tmp ON)
else()
    message(STATUS "Boost fetaures -- not found. set BOOST_ROOT to enable.")
    message(WARNING "Boost is required for command line applications")
endif()
set(TECA_HAS_BOOST ${tmp} CACHE BOOL "Boost features")

# configure for Python
set(tmp OFF)
find_package(PythonInterp)
find_package(PythonLibs)
find_package(Numpy)
find_program(swig_cmd swig)
if ((PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND NUMPY_FOUND) AND ((DEFINED TECA_HAS_PYTHON AND TECA_HAS_PYTHON) OR (NOT DEFINED TECA_HAS_PYTHON)))
    message(STATUS "Python features -- enabled")
    set(tmp ON)
    include(CMake/teca_python.cmake)
else()
    message(STATUS "Python features -- disabled")
endif()
set(TECA_HAS_PYTHON ${tmp} CACHE BOOL "Python binding")

# generaste global configuration
configure_file(teca_config.h.in teca_config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# build library components and applications
add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(io)
add_subdirectory(alg)
add_subdirectory(apps)

# enable regression tests
set(BUILD_TESTING OFF CACHE BOOL "Enable tests")
if (BUILD_TESTING)
    add_subdirectory(test)
endif()
