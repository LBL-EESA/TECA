project(teca_test)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test)

add_subdirectory(test_array)
add_subdirectory(test_amr)
if (TECA_HAS_PYTHON)
    add_subdirectory(python)
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

teca_add_test(test_pipeline_time_average
    SOURCES test_pipeline_time_average.cpp
    LIBS teca_core teca_test_array ${teca_test_link}
    COMMAND test_pipeline_time_average)

teca_add_test(test_pipeline_branches
    SOURCES test_pipeline_branches.cpp
    LIBS teca_core teca_test_array ${teca_test_link}
    COMMAND test_pipeline_branches)

teca_add_test(test_pipeline_temporal_reduction
    SOURCES test_pipeline_temporal_reduction.cpp
    LIBS teca_core teca_test_array ${teca_test_link}
    COMMAND test_pipeline_temporal_reduction)

if (TECA_HAS_NETCDF)
    # TODO -- cmake_parse_arguments eats ""
    add_executable(test_cf_reader test_cf_reader.cpp)
    target_link_libraries(test_cf_reader
        teca_core teca_data teca_io ${teca_test_link})
    if (TECA_DATA_ROOT)
        add_test(NAME test_cf_reader
            COMMAND test_cf_reader
            "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
            "test_cf_reader_%t%.vtk" 1 2 lon lat "" time U850 V850
            WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        set_tests_properties(test_cf_reader
            PROPERTIES FAIL_REGULAR_EXPRESSION "ERROR")
    endif()

    # TODO -- locate test data for this test
    add_executable(test_cartesian_mesh_regrid
        test_cartesian_mesh_regrid.cpp)
    target_link_libraries(test_cartesian_mesh_regrid
        teca_core teca_data teca_io teca_alg ${teca_test_link})

    # TODO -- cmake_parse_arguments eats ""
    add_executable(test_connected_components test_connected_components.cpp)
    target_link_libraries(test_connected_components
        teca_core teca_data teca_io teca_alg ${teca_test_link})
    if (TECA_DATA_ROOT)
        add_test(NAME test_connected_components
            COMMAND test_connected_components
            "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
            lon lat "" time U850 V850 0 0 15 "test_connected_components_%t%.vtk"
            WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        set_tests_properties(test_connected_components
            PROPERTIES FAIL_REGULAR_EXPRESSION "ERROR")
    endif()
endif()

teca_add_test(test_temporal_average
    SOURCES test_temporal_average.cpp create_test_table.cxx
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_temporal_average
    "${TECA_DATA_ROOT}/cam5_1_amip_run2.cam2.h2.1991-10-01-10800.nc"
    test_temporal_average_%t%.vtk 0 -1 3 U850
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_table_writer
    SOURCES test_table_writer.cpp create_test_table.cxx
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_table_writer)

teca_add_test(test_table_reader
    SOURCES test_table_reader.cpp create_test_table.cxx
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_table_reader)

set_tests_properties(test_table_reader PROPERTIES FAIL_REGULAR_EXPRESSION FAIL)

teca_add_test(test_dataset_diff
    SOURCES test_dataset_diff.cpp create_test_table.cxx
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_dataset_diff)

