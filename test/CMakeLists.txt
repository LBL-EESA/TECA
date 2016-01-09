project(teca_test)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test)

add_subdirectory(test_array)
add_subdirectory(test_amr)
if (TECA_HAS_PYTHON)
    add_subdirectory(test_py)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_alg,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_test_array,INTERFACE_INCLUDE_DIRECTORIES>)

set(teca_test_link pthread)

if (TECA_HAS_MPI)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
    list(APPEND teca_test_link ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()

if (TECA_HAS_BOOST)
    include_directories(SYSTEM ${Boost_INCLUDE_DIR})
    list(APPEND teca_test_link ${Boost_LIBRARIES})
endif()

# core tests
add_executable(test_pipeline_time_average
    test_pipeline_time_average.cpp)
target_link_libraries(test_pipeline_time_average
    teca_core teca_test_array ${teca_test_link})

add_executable(test_pipeline_branches
    test_pipeline_branches.cpp)
target_link_libraries(test_pipeline_branches
    teca_core teca_test_array ${teca_test_link})

add_executable(test_pipeline_temporal_reduction
    test_pipeline_temporal_reduction.cpp)
target_link_libraries(test_pipeline_temporal_reduction
    teca_core teca_test_array ${teca_test_link})

# io tests
if (TECA_HAS_NETCDF)
    add_executable(test_cf_reader test_cf_reader.cpp)
    target_link_libraries(test_cf_reader
        teca_core teca_data teca_io ${teca_test_link})
    if (BUILD_TESTING AND TECA_DATA_ROOT)
        add_test(NAME test_cf_reader
            COMMAND test_cf_reader
            "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
            "test_cf_reader_%t%.vtk" 1 2 lon lat "" time U850 V850
            WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endif()

    add_executable(test_temporal_average test_temporal_average.cpp)
    target_link_libraries(test_temporal_average
        teca_core teca_data teca_io teca_alg ${teca_test_link})
    if (BUILD_TESTING AND TECA_DATA_ROOT)
        add_test(NAME test_temporal_average
            COMMAND test_temporal_average
            "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-01-10800\\.nc"
            test_temporal_average_%t%.vtk 0 -1 3 U850
            WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endif()

    add_executable(test_cartesian_mesh_regrid
        test_cartesian_mesh_regrid.cpp)
    target_link_libraries(test_cartesian_mesh_regrid
        teca_core teca_data teca_io teca_alg ${teca_test_link})
    #if (BUILD_TESTING AND TECA_DATA_ROOT)
    #    add_test(NAME test_cartesian_mesh_regrid
    #        COMMAND test_cartesian_mesh_regrid
    #        TODO ARGS
    #        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    #endif()

    add_executable(test_connected_components test_connected_components.cpp)
    target_link_libraries(test_connected_components
        teca_core teca_data teca_io teca_alg ${teca_test_link})
    if (BUILD_TESTING AND TECA_DATA_ROOT)
        add_test(NAME test_connected_components
            COMMAND test_connected_components
            "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-01-10800\\.nc"
            lon lat "" time U850 V850 1 2 15 "test_connected_components_%t%.vtk"
            WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endif()
endif()

add_executable(test_table_writer test_table_writer.cpp)
target_link_libraries(test_table_writer
    teca_core teca_data teca_io teca_alg ${teca_test_link})
if (BUILD_TESTING)
    add_test(NAME test_table_writer
        COMMAND test_table_writer
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()
