cmake_minimum_required(VERSION 2.8)
project(teca)

set(teca_srcs
    teca_algorithm.cxx
    teca_binary_stream.cxx
    teca_meta_data.cxx
    teca_temporal_reduction.cxx
    teca_threaded_algorithm.cxx
    teca_variant_array.cxx
    )

if (MSVC)
    # enable exceptions
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
else()
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(teca ${teca_srcs})

set(BUILD_TESTING OFF CACHE BOOL "Enable tests")
if (BUILD_TESTING)
    add_subdirectory(test)
endif()
