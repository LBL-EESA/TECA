project(teca_test)

add_subdirectory(test_array)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_alg,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_test_array,INTERFACE_INCLUDE_DIRECTORIES>
    )

# core tests
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

# io tests
if (NETCDF_FOUND AND VTK_FOUND)
    add_executable(
        test_cf_reader
        test_cf_reader.cpp
        )
    target_link_libraries(test_cf_reader teca_core teca_data teca_io)
endif()
