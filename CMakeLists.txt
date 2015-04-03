cmake_minimum_required(VERSION 2.8)
project(teca)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(teca_srcs
    teca_algorithm.cxx
    teca_array_collection.cxx
    teca_binary_stream.cxx
    teca_cartesian_mesh.cxx
    teca_file_util.cxx
    teca_metadata.cxx
    teca_mesh.cxx
    teca_parallel_id.cxx
    teca_table.cxx
    teca_temporal_reduction.cxx
    teca_threaded_algorithm.cxx
    teca_uniform_cartesian_mesh.cxx
    teca_variant_array.cxx
    )

if (WIN32)
    list(APPEND teca_srcs win_windirent.cxx)
endif()

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
    "${CMAKE_CURRENT_SOURCE_DIR}/teca_platform_tests.cpp"
    CMAKE_FLAGS ${CMAKE_CXX_FLAGS}
    COMPILE_DEFINITIONS "-DCXX11_REGEX_TEST")
if (NOT (CXX11_REGEX_COMPILES AND CXX11_REGEX_WORKS))
    message(FATAL_ERROR "Error: c++11 regex is not supported by this compiler.")
else()
    message(STATUS "Check for c++ regex support -- works")
endif()

find_package(MPI)
if (MPI_C_FOUND)
    include_directories(${MPI_C_INCLUDE_PATH})
    add_definitions(-DTECA_MPI)
endif()

find_package(NetCDF)
if (NETCDF_FOUND)
    list(APPEND teca_srcs teca_cf_reader.cxx)
    include_directories(${NETCDF_INCLUDE_DIRS})
endif()

add_library(teca ${teca_srcs})

if (MPI_CXX_FOUND)
    target_link_libraries(teca ${MPI_CXX_LIBRARIES})
endif()

if (NETCDF_FOUND)
    target_link_libraries(teca ${NETCDF_LIBRARIES})
endif()

set(BUILD_TESTING OFF CACHE BOOL "Enable tests")
if (BUILD_TESTING)
    add_subdirectory(test)
endif()
