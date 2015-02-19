cmake_minimum_required(VERSION 2.8)
project(teca)

find_package(MPI)

set(teca_srcs
    teca_algorithm.cxx
    teca_binary_stream.cxx
    teca_metadata.cxx
    teca_parallel_id.cxx
    teca_temporal_reduction.cxx
    teca_threaded_algorithm.cxx
    teca_variant_array.cxx
    )

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

if (MPI_C_FOUND)
    include_directories(${MPI_C_INCLUDE_PATH})
    add_definitions(-DTECA_MPI)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(teca ${teca_srcs})

if (MPI_CXX_FOUND)
    target_link_libraries(teca ${MPI_CXX_LIBRARIES})
endif()

set(BUILD_TESTING OFF CACHE BOOL "Enable tests")
if (BUILD_TESTING)
    add_subdirectory(test)
endif()
