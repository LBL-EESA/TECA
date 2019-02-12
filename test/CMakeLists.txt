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
    $<TARGET_PROPERTY:teca_system,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_test_array,INTERFACE_INCLUDE_DIRECTORIES>)

set(teca_test_link pthread teca_system)

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

teca_add_test(test_stack_trace_signal_handler
    SOURCES test_stack_trace_signal_handler.cpp
    LIBS ${teca_test_link}
    COMMAND test_stack_trace_signal_handler 8 0)

teca_add_test(test_cf_reader
    SOURCES test_cf_reader.cpp
    LIBS teca_core teca_data teca_io ${teca_test_link}
    COMMAND test_cf_reader
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    "test_cf_reader_%t%.%e%" 1 2 lon lat "." time U850 V850
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_connected_components
    SOURCES test_connected_components.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_connected_components
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    lon lat "." time U850 V850 0 0 15 "test_connected_components_%t%.%e%"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cartesian_mesh_regrid
    SOURCES test_cartesian_mesh_regrid.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_cartesian_mesh_regrid
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-01-10800\\.nc"
    lon lat "." time 1 PSL "${TECA_DATA_ROOT}/landsea\\.nc" lon lat "." "." 1 LSMASK
    test_cartesian_mesh_regrid.%e% 0 0 1 359 -89 89 0 0
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_temporal_average
    SOURCES test_temporal_average.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_temporal_average
    "${TECA_DATA_ROOT}/cam5_1_amip_run2.cam2.h2.1991-10-01-10800.nc"
    test_temporal_average_%t%.%e% 0 -1 3 U850
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_table_writer
    SOURCES test_table_writer.cpp teca_test_util.cxx
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_table_writer)

teca_add_test(test_table_reader
    SOURCES test_table_reader.cpp teca_test_util.cxx
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_table_reader)

teca_add_test(test_dataset_diff
    SOURCES test_dataset_diff.cpp teca_test_util.cxx
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_dataset_diff)

teca_add_test(test_table_sort
    SOURCES test_table_sort.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_table_sort "${TECA_DATA_ROOT}/cam5_temperature_stats.bin"
    "${TECA_DATA_ROOT}/test_table_sort.bin" step
    REQ_TECA_DATA)

teca_add_test(test_descriptive_statistics_serial
    EXEC_NAME test_descriptive_statistics
    SOURCES test_descriptive_statistics.cpp teca_test_util.cxx
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_descriptive_statistics
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    "${TECA_DATA_ROOT}/test_descriptive_statistics.bin" 0 -1 1 TMQ T200 T500
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_descriptive_statistics_threads
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_descriptive_statistics
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    "${TECA_DATA_ROOT}/test_descriptive_statistics.bin" 0 -1 ${TECA_TEST_CORES}
    TMQ T200 T500
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_descriptive_statistics_mpi
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND ${MPIEXEC} -n ${TECA_TEST_CORES} test_descriptive_statistics
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    "${TECA_DATA_ROOT}/test_descriptive_statistics.bin" 0 -1 1 TMQ T200 T500
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_descriptive_statistics_mpi_threads
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND ${MPIEXEC} -n ${HALF_CORES} test_descriptive_statistics
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    "${TECA_DATA_ROOT}/test_descriptive_statistics.bin" 0 -1 ${HALF_CORES}
    TMQ T200 T500
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_binary_stream
    SOURCES test_binary_stream.cpp
    LIBS teca_core teca_data teca_alg ${teca_test_link}
    COMMAND test_binary_stream)

teca_add_test(test_binary_stream_mpi
    SOURCES test_binary_stream.cpp
    LIBS teca_core teca_data teca_alg ${teca_test_link}
    COMMAND ${MPIEXEC} -n 2 test_binary_stream
    FEATURES ${TECA_HAS_MPI})

teca_add_test(test_tc_candidates_serial
    COMMAND test_tc_candidates
    "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin" 0 3 1
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(test_tc_candidates_mpi
    COMMAND ${MPIEXEC} -n ${TECA_TEST_CORES} test_tc_candidates
    "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin" 0 3 1
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_tc_candidates_threads
    EXEC_NAME test_tc_candidates
    SOURCES test_tc_candidates.cpp teca_test_util.cxx
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_tc_candidates
    "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin" 0 3 ${TECA_TEST_CORES}
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(test_tc_candidates_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_CORES} test_tc_candidates
    "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin" 0 3 ${HALF_CORES}
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_tc_trajectory
    EXEC_NAME test_tc_trajectory
    SOURCES test_tc_trajectory.cpp teca_test_util.cxx
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_tc_trajectory
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_candidates_2005_09.bin"
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_tracks_2005_09.bin" 1600 17 2
    FEATURES ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(test_tc_classify
    SOURCES test_tc_classify.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_tc_classify
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_tracks_2005_09.bin"
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_classify_2005_09.bin"
    REQ_TECA_DATA)

teca_add_test(test_table_reader_distribute_serial
    COMMAND test_table_reader_distribute
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin"
    "${TECA_DATA_ROOT}/test_table_reader_distribute_20.bin"
    "step" 0 -1 1
    REQ_TECA_DATA)

teca_add_test(test_table_reader_distribute_mpi
    COMMAND ${MPIEXEC} -n ${TECA_TEST_CORES} test_table_reader_distribute
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin"
    "${TECA_DATA_ROOT}/test_table_reader_distribute_20.bin"
    "step" 0 -1 1
    FEATURES ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_table_reader_distribute_threads
    EXEC_NAME test_table_reader_distribute
    SOURCES test_table_reader_distribute.cpp teca_test_util.cxx
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_table_reader_distribute
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin"
    "${TECA_DATA_ROOT}/test_table_reader_distribute_20.bin"
    "step" 0 -1 ${TECA_TEST_CORES}
    REQ_TECA_DATA)

teca_add_test(test_table_reader_distribute_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_CORES} test_table_reader_distribute
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin"
    "${TECA_DATA_ROOT}/test_table_reader_distribute_20.bin"
    "step" 0 -1 ${HALF_CORES}
    FEATURES ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_serial
    EXEC_NAME test_tc_wind_radii
    SOURCES test_tc_wind_radii.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_tc_wind_radii "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800.bin"
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_1990s/.*\\.nc$" "${TECA_DATA_ROOT}/test_tc_wind_radii.bin"
    "!(((track_id==4)&&(surface_wind*3.6d>=177.0d))||((track_id==191)&&(surface_wind*3.6d>=249.0d))||((track_id==523)&&(3.6d*surface_wind>=209.0d)))"
    "32" "1" "1" "0" "-1"
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_mpi
    COMMAND ${MPIEXEC} -n ${TECA_TEST_CORES} test_tc_wind_radii "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800.bin"
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_1990s/.*\\.nc$" "${TECA_DATA_ROOT}/test_tc_wind_radii.bin"
    "!(((track_id==4)&&(surface_wind*3.6d>=177.0d))||((track_id==191)&&(surface_wind*3.6d>=249.0d))||((track_id==523)&&(3.6d*surface_wind>=209.0d)))"
    "32" "1" "1" "0" "-1"
    FEATURES ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_CORES} test_tc_wind_radii "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800.bin"
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_1990s/.*\\.nc$" "${TECA_DATA_ROOT}/test_tc_wind_radii.bin"
    "!(((track_id==4)&&(surface_wind*3.6d>=177.0d))||((track_id==191)&&(surface_wind*3.6d>=249.0d))||((track_id==523)&&(3.6d*surface_wind>=209.0d)))"
    "32" "1" ${HALF_CORES} "0" "-1"
    FEATURES ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_threads
    COMMAND test_tc_wind_radii "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800.bin"
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_1990s/.*\\.nc$" "${TECA_DATA_ROOT}/test_tc_wind_radii.bin"
    "!(((track_id==4)&&(surface_wind*3.6d>=177.0d))||((track_id==191)&&(surface_wind*3.6d>=249.0d))||((track_id==523)&&(3.6d*surface_wind>=209.0d)))"
    "32" "1" ${TECA_TEST_CORES} "0" "-1"
    REQ_TECA_DATA)

teca_add_test(test_type_select
    SOURCES test_type_select.cpp
    LIBS teca_core teca_alg ${teca_test_link}
    COMMAND test_type_select)

teca_add_test(test_parser
    SOURCES test_parser.cpp
    LIBS teca_core teca_alg ${teca_test_link}
    COMMAND test_parser)

teca_add_test(test_variant_array_operator
    SOURCES test_variant_array_operator.cpp
    LIBS teca_core teca_alg ${teca_test_link}
    COMMAND test_variant_array_operator)

teca_add_test(test_evaluate_expression_table
    SOURCES test_evaluate_expression_table.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_evaluate_expression_table "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800.bin"
    "${TECA_DATA_ROOT}/test_evaluate_expression_table.bin"
    "!((track_id==191)&&((surface_wind*3.6d>=209.0d)&&(surface_wind*3.6d<249.0d)))"
    REQ_TECA_DATA)

teca_add_test(test_table_remove_rows_track_id
    EXEC_NAME test_table_remove_rows
    SOURCES test_table_remove_rows.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_table_remove_rows "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800.bin"
    "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800_median_in_cat.bin"
    "(track_id==488)||(track_id==186)||(track_id==578)||(track_id==4)||(track_id==523)||(track_id==191)"
    REQ_TECA_DATA)

teca_add_test(test_table_remove_rows_cat_4
    COMMAND test_table_remove_rows "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800.bin"
    "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800_cat_4.bin"
    "!((surface_wind*3.6d>=209.0d)&&(surface_wind*3.6d<249.0d))"
    REQ_TECA_DATA)

teca_add_test(test_table_region_mask
    SOURCES test_table_region_mask.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_table_region_mask "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800.bin"
    "${TECA_DATA_ROOT}/test_table_region_mask.bin" "N Atlantic"
    REQ_TECA_DATA)

teca_add_test(test_event_filter
    SOURCES test_event_filter.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_event_filter "${TECA_DATA_ROOT}/test_tc_candidates_20.bin"
    "${TECA_DATA_ROOT}/test_event_filter.bin"
    REQ_TECA_DATA)
