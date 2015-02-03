cmake_minimum_required(VERSION 2.8)
project(teca_test)

set(array_test_srcs
    array.cxx
    array_add.cxx
    array_executive.cxx
    array_scalar_multiply.cxx
    array_source.cxx
    array_temporal_stats.cxx
    array_time_average.cxx
    array_writer.cxx
    )

add_library(teca_test_array ${array_test_srcs})
target_link_libraries(teca_test_array teca)

add_executable(
    test_array_time_average
    test_array_time_average.cpp
    )

target_link_libraries(
    test_array_time_average
    teca teca_test_array
    )

add_executable(
    test_array_branches
    test_array_branches.cpp
    )
target_link_libraries(
    test_array_branches
    teca teca_test_array
    )

if (MPI_C_FOUND) 
add_executable(
    test_array_temporal_reduction
    test_array_temporal_reduction.cpp
    )
target_link_libraries(
    test_array_temporal_reduction
    teca teca_test_array
    ${MPI_CXX_LIBRARIES}
    )
endif()
