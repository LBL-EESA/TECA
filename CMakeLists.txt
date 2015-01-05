cmake_minimum_required(VERSION 2.8)
project(teca)

set(teca_srcs
    teca_meta_data.cxx
    teca_algorithm.cxx
    )

if (MSVC)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(teca ${teca_srcs})
