cmake_minimum_required(VERSION 2.8)

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
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
    message(FATAL_ERROR "Error: c++11 regex is not supported by this compiler.")
else()
    message(STATUS "Check for c++ regex support -- works")
endif()

# locate dependencies
find_package(MPI)
if (MPI_C_FOUND)
    message(STATUS "MPI features -- enabled")
    set(TECA_HAS_MPI ON CACHE INTERNAL "MPI features")
else()
    message(STATUS "MPI features -- not found.")
endif()

find_package(NetCDF)
if (NETCDF_FOUND)
    message(STATUS "NetCDF features -- enabled")
    set(TECA_HAS_NETCDF ON CACHE INTERNAL "NetCDF features")
else()
    message(STATUS "NetCDF features -- not found. set NETCDF_DIR to enable.")
endif()

find_package(VTK QUIET)
if (VTK_FOUND)
    message(STATUS "VTK features -- enabled")
    set(TECA_HAS_VTK ON CACHE INTERNAL "VTK features")
else()
    message(STATUS "VTK features -- not found. set VTK_DIR to enable.")
endif()

find_package(Boost QUIET COMPONENTS program_options)
if (Boost_FOUND)
    message(STATUS "Boost features -- enabled")
    set(TECA_HAS_BOOST ON CACHE INTERNAL "Boost features")
else()
    message(STATUS "Boost fetaures -- not found. set BOOST_ROOT to enable.")
endif()

# build library components and applications
add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(io)
add_subdirectory(alg)

# enable regression tests
set(BUILD_TESTING OFF CACHE BOOL "Enable tests")
if (BUILD_TESTING)
    add_subdirectory(test)
endif()
