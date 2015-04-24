cmake_minimum_required(VERSION 2.8)
project(teca_core)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(teca_core_srcs
    teca_algorithm.cxx
    teca_binary_stream.cxx
    teca_metadata.cxx
    teca_parallel_id.cxx
    teca_temporal_reduction.cxx
    teca_threaded_algorithm.cxx
    teca_time_step_executive.cxx
    teca_variant_array.cxx
    )

find_package(MPI)
if (MPI_C_FOUND)
    include_directories(${MPI_C_INCLUDE_PATH})
    add_definitions(-DTECA_MPI)
endif()

add_library(teca_core ${teca_core_srcs})

if (MPI_CXX_FOUND)
    target_link_libraries(teca_core ${MPI_CXX_LIBRARIES})
endif()

target_include_directories(teca_core
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )
