project(teca_core_test)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    )

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
target_link_libraries(teca_test_array teca_core)

add_executable(
    test_array_time_average
    test_array_time_average.cpp
    )
target_link_libraries(
    test_array_time_average
    teca_core teca_test_array
    )

add_executable(
    test_array_branches
    test_array_branches.cpp
    )
target_link_libraries(
    test_array_branches
    teca_core teca_test_array
    )

add_executable(
    test_array_temporal_reduction
    test_array_temporal_reduction.cpp
    )
target_link_libraries(
    test_array_temporal_reduction
    teca_core teca_test_array
    )
if (MPI_CXX_FOUND)
    target_link_libraries(
        test_array_temporal_reduction
        ${MPI_CXX_LIBRARIES}
        )
endif()
