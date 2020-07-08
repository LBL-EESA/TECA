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

teca_add_test(test_pipeline_index_reduce
    SOURCES test_pipeline_index_reduce.cpp
    LIBS teca_core teca_test_array ${teca_test_link}
    COMMAND test_pipeline_index_reduce)

teca_add_test(test_stack_trace_signal_handler
    SOURCES test_stack_trace_signal_handler.cpp
    LIBS ${teca_test_link}
    COMMAND test_stack_trace_signal_handler 8 0)

teca_add_test(test_mpi_error_handler
    SOURCES test_mpi_error_handler.cpp
    LIBS ${teca_test_link}
    COMMAND test_mpi_error_handler
    FEATURES ${TECA_HAS_MPI}
    WILL_FAIL)

teca_add_test(test_cf_reader_cam5
    EXEC_NAME test_cf_reader
    SOURCES test_cf_reader.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_cf_reader
    -i "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    -o "test_cf_reader_cam5_%t%.%e%" -s 1,2 -x lon -y lat -t time U850 V850
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_reader_cfsr
    COMMAND test_cf_reader
    -i "${TECA_DATA_ROOT}/NCEP_CFSR_0\\.5_1979\\.nc"
    -o "test_cf_reader_cfsr_%t%.%e%" -s 1,2 -x longitude -y latitude
    -b 65,110,10,55,0,0 elevation
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_reader_era5
    COMMAND test_cf_reader
    -i "${TECA_DATA_ROOT}/e5\.oper\.an\.vinteg\.162_072_viwvn.*\.nc"
    -o "test_cf_reader_era5_%t%.%e%" -s 0,-1 -x longitude -y latitude -t time VIWVN
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_reader_file_time
    COMMAND test_cf_reader
    -i "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_20170210_06\.nc"
    -o "test_cf_reader_file_time_%t%.%e%" -s 0,-1 -x lon -y lat
    -n "ARTMIP_MERRA_2D_%Y%m%d_%H.nc" IVT
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_cam5_serial
    EXEC_NAME test_cf_writer
    SOURCES test_cf_writer.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_cf_writer
    -i "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    -o "test_cf_writer_cam5_s_%t%.nc" -s 0,-1 -x lon -y lat -t time -c 2 -n 1 U850 V850
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_cfsr_serial
    COMMAND test_cf_writer
    -i "${TECA_DATA_ROOT}/NCEP_CFSR_0\\.5_1979\\.nc"
    -o "test_cf_writer_NCEP_CFSR_s_%t%.nc" -s 0,-1 -x longitude -y latitude
    -b 65,110,10,55,0,0 -c 2 -n 1 elevation
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_era5_serial
    COMMAND test_cf_writer
    -i "${TECA_DATA_ROOT}/e5\.oper\.an\.vinteg\.162_072_viwvn.*\.nc"
    -o "test_cf_writer_era5_s_%t%.nc" -s 0,-1 -x longitude -y latitude -t time -c 2 -n 1 VIWVN
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_cam5_threads
    COMMAND test_cf_writer
    -i "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    -o "test_cf_writer_cam5_t_%t%.nc" -s 0,-1 -x lon -y lat -t time -c 2 -n ${TEST_CORES} U850 V850
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_cfsr_threads
    COMMAND test_cf_writer
    -i "${TECA_DATA_ROOT}/NCEP_CFSR_0\\.5_1979\\.nc"
    -o "test_cf_writer_NCEP_CFSR_t_%t%.nc" -s 0,-1 -x longitude -y latitude
    -b 65,110,10,55,0,0 -n ${TEST_CORES} -c 2 elevation
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_era5_threads
    COMMAND test_cf_writer
    -i "${TECA_DATA_ROOT}/e5\.oper\.an\.vinteg\.162_072_viwvn.*\.nc"
    -o "test_cf_writer_era5_t_%t%.nc" -s 0,-1 -x longitude -y latitude -t time -c 2 -n ${TEST_CORES} VIWVN
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_cam5_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_writer
    -i "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    -o "test_cf_writer_cam5_m_%t%.nc" -s 0,-1 -x lon -y lat -t time -c 2 -n 1 U850 V850
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_cfsr_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_writer
    -i "${TECA_DATA_ROOT}/NCEP_CFSR_0\\.5_1979\\.nc"
    -o "test_cf_writer_NCEP_CFSR_m_%t%.nc" -s 0,-1 -x longitude -y latitude
    -b 65,110,10,55,0,0 -c 2 -n 1 elevation
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_era5_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_writer
    -i "${TECA_DATA_ROOT}/e5\.oper\.an\.vinteg\.162_072_viwvn.*\.nc"
    -o "test_cf_writer_era5_m_%t%.nc" -s 0,-1 -x longitude -y latitude -t time -c 2 -n 1 VIWVN
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_cam5_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_cf_writer
    -i "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    -o "test_cf_writer_cam5_mt_%t%.nc" -s 0,-1 -x lon -y lat -t time -c 1 -n 2 U850 V850
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_cfsr_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_cf_writer
    -i "${TECA_DATA_ROOT}/NCEP_CFSR_0\\.5_1979\\.nc"
    -o "test_cf_writer_NCEP_CFSR_mt_%t%.nc" -s 0,-1 -x longitude -y latitude
    -b 65,110,10,55,0,0 -c 1 -n 2 elevation
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_era5_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_cf_writer
    -i "${TECA_DATA_ROOT}/e5\.oper\.an\.vinteg\.162_072_viwvn.*\.nc"
    -o "test_cf_writer_era5_mt_%t%.nc" -s 0,-1 -x longitude -y latitude -t time -c 1 -n 2 VIWVN
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_connected_components
    SOURCES test_connected_components.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_connected_components
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    lon lat "." time U850 V850 0 0 15 "test_connected_components_%t%.%e%"
    "${TECA_DATA_ROOT}/test_connected_components.bin"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_2d_component_area
    SOURCES test_2d_component_area.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_2d_component_area 720 360 0 8 8 0
    "test_2d_component_area_%t%.%e%")

teca_add_test(test_2d_component_area_contiguous_labels
    COMMAND test_2d_component_area 720 360 0 8 8 1
    "test_2d_component_area_contiguous_labels_%t%.%e%")

teca_add_test(test_2d_component_area_negative_area
    COMMAND test_2d_component_area 720 360 1 8 8 1
    "test_2d_component_area_contiguous_labels_%t%.%e%")

teca_add_test(test_component_area_filter
    SOURCES test_component_area_filter.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_component_area_filter 720 360 8 8 10000000 0
    "test_component_area_filter_%t%.%e%")

teca_add_test(test_component_area_filter_contiguous_labels
    COMMAND test_component_area_filter 720 360 8 8 10000000 1
    "test_component_area_filter_%t%.%e%")

teca_add_test(test_cartesian_mesh_regrid
    SOURCES test_cartesian_mesh_regrid.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_cartesian_mesh_regrid
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-01-10800\\.nc"
    lon lat "." time 1 PSL "${TECA_DATA_ROOT}/landsea\\.nc" lon lat "." "." 1 LSMASK
    "${TECA_DATA_ROOT}/test_cartesian_mesh_regrid.bin" 0 0 1 359 -89 89 0 0
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_latitude_damper
    SOURCES test_latitude_damper.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_latitude_damper 361 181 10 "test_latitude_damper_%t%.%e%")

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

teca_add_test(test_dataset_diff_same
    EXEC_NAME test_dataset_diff
    SOURCES test_dataset_diff.cpp teca_test_util.cxx
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_dataset_diff 1)

teca_add_test(test_dataset_diff_different
    COMMAND test_dataset_diff 0
    WILL_FAIL)

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
    COMMAND test_descriptive_statistics 0
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    "${TECA_DATA_ROOT}/test_descriptive_statistics.bin" 0 -1 1 TMQ T200 T500
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_descriptive_statistics_era5
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_descriptive_statistics 4
    "${TECA_DATA_ROOT}/cam5_1_amip_run2.cam2.h2.1991-10-01-06Z.nc"
    "${TECA_DATA_ROOT}/cam5_1_amip_run2.cam2.h2.1991-10-01-18Z.nc"
    "${TECA_DATA_ROOT}/cam5_1_amip_run2.cam2.h2.1991-10-02-06Z.nc"
    "${TECA_DATA_ROOT}/cam5_1_amip_run2.cam2.h2.1991-10-02-18Z.nc"
    4653.25 4653.75 4654.25 4654.75
    "${TECA_DATA_ROOT}/test_descriptive_statistics.bin" 0 -1 1 TMQ T200 T500
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_descriptive_statistics_threads
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_descriptive_statistics 0
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    "${TECA_DATA_ROOT}/test_descriptive_statistics.bin" 0 -1 ${TEST_CORES}
    TMQ T200 T500
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_descriptive_statistics_mpi
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_descriptive_statistics 0
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    "${TECA_DATA_ROOT}/test_descriptive_statistics.bin" 0 -1 1 TMQ T200 T500
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_descriptive_statistics_mpi_threads
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_descriptive_statistics 0
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    "${TECA_DATA_ROOT}/test_descriptive_statistics.bin" 0 -1 2
    TMQ T200 T500
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_streaming_reduce_threads
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_descriptive_statistics 0
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3.*\\.nc"
    "${TECA_DATA_ROOT}/test_streaming_reduce.bin" 0 -1 ${TWICE_TEST_CORES}
    prw
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_streaming_reduce_mpi_threads
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_descriptive_statistics 0
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3.*\\.nc"
    "${TECA_DATA_ROOT}/test_streaming_reduce.bin" 0 -1 4
    prw
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF}
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
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_tc_candidates
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
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin" 0 3 ${TEST_CORES}
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(test_tc_candidates_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_tc_candidates
    "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin" 0 3 2
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
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_table_reader_distribute
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
    "step" 0 -1 ${TEST_CORES}
    REQ_TECA_DATA)

teca_add_test(test_table_reader_distribute_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_table_reader_distribute
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin"
    "${TECA_DATA_ROOT}/test_table_reader_distribute_20.bin"
    "step" 0 -1 2
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
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_tc_wind_radii "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800.bin"
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_1990s/.*\\.nc$" "${TECA_DATA_ROOT}/test_tc_wind_radii.bin"
    "!(((track_id==4)&&(surface_wind*3.6d>=177.0d))||((track_id==191)&&(surface_wind*3.6d>=249.0d))||((track_id==523)&&(3.6d*surface_wind>=209.0d)))"
    "32" "1" "1" "0" "-1"
    FEATURES ${TECA_HAS_MPI} ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_tc_wind_radii "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800.bin"
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_1990s/.*\\.nc$" "${TECA_DATA_ROOT}/test_tc_wind_radii.bin"
    "!(((track_id==4)&&(surface_wind*3.6d>=177.0d))||((track_id==191)&&(surface_wind*3.6d>=249.0d))||((track_id==523)&&(3.6d*surface_wind>=209.0d)))"
    32 1 2 0 -1
    FEATURES (${TECA_HAS_MPI} AND ${TECA_HAS_NETCDF})
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_threads
    COMMAND test_tc_wind_radii "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800.bin"
    "${TECA_DATA_ROOT}/cam5_1_amip_run2_1990s/.*\\.nc$" "${TECA_DATA_ROOT}/test_tc_wind_radii.bin"
    "!(((track_id==4)&&(surface_wind*3.6d>=177.0d))||((track_id==191)&&(surface_wind*3.6d>=249.0d))||((track_id==523)&&(3.6d*surface_wind>=209.0d)))"
    32 1 ${TEST_CORES} 0 -1
    FEATURES ${TECA_HAS_NETCDF}
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

teca_add_test(test_binary_segmentation
    SOURCES test_binary_segmentation.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_binary_segmentation "${TECA_DATA_ROOT}/prw_hus_day_MRI.*\\.nc$"
    prw 50 75 "prw_segmentation_50-75_%t%.%e%"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_serial
    EXEC_NAME test_bayesian_ar_detect
    SOURCES test_bayesian_ar_detect.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_bayesian_ar_detect
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_%t%.nc" 1 0 4
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_threads
    COMMAND test_bayesian_ar_detect
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_%t%.nc" ${TEST_CORES} 0 4
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_bayesian_ar_detect
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_%t%.nc" -1 0 4
    FEATURES (${TECA_HAS_MPI} AND ${TECA_HAS_NETCDF})
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_pass_through
    EXEC_NAME test_normalize_coordinates
    SOURCES test_normalize_coordinates.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_normalize_coordinates 90 45 7 0 0 0 0 360 -90 90 0 10
    "${TECA_DATA_ROOT}/test_normalize_coordinates.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_x
    COMMAND test_normalize_coordinates 90 45 7 1 0 0 0 360 -90 90 0 10
    "${TECA_DATA_ROOT}/test_normalize_coordinates.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_y
    COMMAND test_normalize_coordinates 90 45 7 0 1 0 0 360 -90 90 0 10
    "${TECA_DATA_ROOT}/test_normalize_coordinates.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_z
    COMMAND test_normalize_coordinates 90 45 7 0 0 1 0 360 -90 90 0 10
    "${TECA_DATA_ROOT}/test_normalize_coordinates.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_xy
    COMMAND test_normalize_coordinates 90 45 7 1 1 0 0 360 -90 90 0 10
    "${TECA_DATA_ROOT}/test_normalize_coordinates.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_xz
    COMMAND test_normalize_coordinates 90 45 7 1 0 1 0 360 -90 90 0 10
    "${TECA_DATA_ROOT}/test_normalize_coordinates.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_yz
    COMMAND test_normalize_coordinates 90 45 7 0 1 1 0 360 -90 90 0 10
    "${TECA_DATA_ROOT}/test_normalize_coordinates.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_xyz
    COMMAND test_normalize_coordinates 90 45 7 1 1 1 0 360 -90 90 0 10
    "${TECA_DATA_ROOT}/test_normalize_coordinates.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_pass_through_subset
    COMMAND test_normalize_coordinates 90 45 7 0 0 0 40 190 -30 45 3 7
    "${TECA_DATA_ROOT}/test_normalize_coordinates_subset.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_x_subset
    COMMAND test_normalize_coordinates 90 45 7 1 0 0 40 190 -30 45 3 7
    "${TECA_DATA_ROOT}/test_normalize_coordinates_subset.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_y_subset
    COMMAND test_normalize_coordinates 90 45 7 0 1 0 40 190 -30 45 3 7
    "${TECA_DATA_ROOT}/test_normalize_coordinates_subset.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_z_subset
    COMMAND test_normalize_coordinates 90 45 7 0 0 1 40 190 -30 45 3 7
    "${TECA_DATA_ROOT}/test_normalize_coordinates_subset.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_xy_subset
    COMMAND test_normalize_coordinates 90 45 7 1 1 0 40 190 -30 45 3 7
    "${TECA_DATA_ROOT}/test_normalize_coordinates_subset.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_xz_subset
    COMMAND test_normalize_coordinates 90 45 7 1 0 1 40 190 -30 45 3 7
    "${TECA_DATA_ROOT}/test_normalize_coordinates_subset.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_yz_subset
    COMMAND test_normalize_coordinates 90 45 7 0 1 1 40 190 -30 45 3 7
    "${TECA_DATA_ROOT}/test_normalize_coordinates_subset.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_xyz_subset
    COMMAND test_normalize_coordinates 90 45 7 1 1 1 40 190 -30 45 3 7
    "${TECA_DATA_ROOT}/test_normalize_coordinates_subset.bin"
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_collective_serial
    EXEC_NAME test_cf_writer_collective
    SOURCES test_cf_writer_collective.cpp
    LIBS teca_core teca_data teca_alg teca_io ${teca_test_link}
    COMMAND test_cf_writer_collective 128 512 128 1
    "${TECA_DATA_ROOT}/test_cf_writer_collective_%t%.bin" 213
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_collective_threads
    COMMAND test_cf_writer_collective 128 512 128 ${TEST_CORES}
    "${TECA_DATA_ROOT}/test_cf_writer_collective_%t%.bin" 213
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_collective_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_writer_collective 128 512 128 1
    "${TECA_DATA_ROOT}/test_cf_writer_collective_%t%.bin" 213
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_collective_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_cf_writer_collective 128 512 128 2
    "${TECA_DATA_ROOT}/test_cf_writer_collective_%t%.bin" 213
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)
