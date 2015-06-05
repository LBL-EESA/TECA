project(teca_test)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test)

add_subdirectory(test_array)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_alg,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_test_array,INTERFACE_INCLUDE_DIRECTORIES>
    )

if (TECA_HAS_MPI)
    include_directories(${MPI_C_INCLUDE_PATH})
    add_definitions(-DTECA_HAS_MPI)
endif()

# core tests
add_executable(
    test_pipeline_time_average
    test_pipeline_time_average.cpp
    )
target_link_libraries(
    test_pipeline_time_average
    teca_core teca_test_array
    )

add_executable(
    test_pipeline_branches
    test_pipeline_branches.cpp
    )
target_link_libraries(
    test_pipeline_branches
    teca_core teca_test_array
    )

add_executable(
    test_pipeline_temporal_reduction
    test_pipeline_temporal_reduction.cpp
    )
target_link_libraries(
    test_pipeline_temporal_reduction
    teca_core teca_test_array
    )
if (TECA_HAS_MPI)
    target_link_libraries(
        test_pipeline_temporal_reduction
        ${MPI_CXX_LIBRARIES}
        )
endif()

# io tests
if (TECA_HAS_NETCDF)
    if (TECA_HAS_VTK)
        add_executable(
            test_cf_reader
            test_cf_reader.cpp
            )
        target_link_libraries(
            test_cf_reader
            teca_core teca_data teca_io
            )
        add_executable(
            test_temporal_average
            test_temporal_average.cpp
            )
        target_link_libraries(
            test_temporal_average
            teca_core teca_data teca_io teca_alg
            )
        add_executable(
            test_cartesian_mesh_regrid
            test_cartesian_mesh_regrid.cpp
            )
        target_link_libraries(
            test_cartesian_mesh_regrid
            teca_core teca_data teca_io teca_alg
            )
    endif()
endif()

add_executable(
    test_table_writer
    test_table_writer.cpp
    )
target_link_libraries(
    test_table_writer
    teca_core teca_data teca_io teca_alg
    )
