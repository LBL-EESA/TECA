project(teca_test)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test)

add_subdirectory(apps)
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

if (TECA_HAS_DATA)
    configure_file(${TECA_DATA_ROOT}/ECMWF-IFS-HR-SST-present.mcf.in
         ${CMAKE_CURRENT_BINARY_DIR}/ECMWF-IFS-HR-SST-present.mcf)

    configure_file(${TECA_DATA_ROOT}/HighResMIP_TC_test.mcf.in
         ${CMAKE_CURRENT_BINARY_DIR}/HighResMIP_TC_test.mcf)

    configure_file(${TECA_DATA_ROOT}/ERA5_lapse_rate_test.mcf.in
         ${CMAKE_CURRENT_BINARY_DIR}/ERA5_lapse_rate_test.mcf)
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

teca_add_test(test_wrf_reader_3d
    EXEC_NAME test_wrf_reader
    SOURCES test_wrf_reader.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_wrf_reader
    -i "${TECA_DATA_ROOT}/wrfout_d01_2020-03-01_crop_00_00_00"
    -o "test_wrf_reader_3d_%t%.vtk" QVAPOR P_HYD U V W T P
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_wrf_reader_2d
    COMMAND test_wrf_reader
    -i "${TECA_DATA_ROOT}/wrfout_d01_2020-03-01_crop_00_00_00"
    -e 0,373,0,678,0,0 -o "test_wrf_reader_2d_%t%.vtk"
    XLAND PSFC QVAPOR P_HYD U V W T P
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
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_cfsr_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_cf_writer
    -i "${TECA_DATA_ROOT}/NCEP_CFSR_0\\.5_1979\\.nc"
    -o "test_cf_writer_NCEP_CFSR_mt_%t%.nc" -s 0,-1 -x longitude -y latitude
    -b 65,110,10,55,0,0 -c 1 -n 2 elevation
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_era5_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_cf_writer
    -i "${TECA_DATA_ROOT}/e5\.oper\.an\.vinteg\.162_072_viwvn.*\.nc"
    -o "test_cf_writer_era5_mt_%t%.nc" -s 0,-1 -x longitude -y latitude -t time -c 1 -n 2 VIWVN
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
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
    "test_2d_component_area_%t%.bin")

teca_add_test(test_2d_component_area_contiguous_labels
    COMMAND test_2d_component_area 720 360 0 8 8 1
    "test_2d_component_area_contiguous_labels_%t%.bin")

teca_add_test(test_2d_component_area_negative_area
    COMMAND test_2d_component_area 720 360 1 8 8 1
    "test_2d_component_area_contiguous_labels_%t%.bin")

teca_add_test(test_component_area_filter
    SOURCES test_component_area_filter.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_component_area_filter 720 360 8 8 10000000 0
    "test_component_area_filter_%t%.bin")

teca_add_test(test_component_area_filter_contiguous_labels
    COMMAND test_component_area_filter 720 360 8 8 10000000 1
    "test_component_area_filter_%t%.bin")

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

teca_add_test(test_simple_moving_average
    SOURCES test_simple_moving_average.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_simple_moving_average
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
    "${TECA_DATA_ROOT}/test_temporal_average" 0 -1 3 1 prw
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
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_test(test_streaming_reduce_threads
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_descriptive_statistics 0
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3.historical_r1i1p1_19500101-19501231\\.nc"
    "${TECA_DATA_ROOT}/test_streaming_reduce.bin" 0 -1 ${TWICE_TEST_CORES}
    prw
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_streaming_reduce_mpi_threads
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_descriptive_statistics 0
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3.historical_r1i1p1_19500101-19501231\\.nc"
    "${TECA_DATA_ROOT}/test_streaming_reduce.bin" 0 -1 4 prw
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_NETCDF} ${TEST_MPI_THREADS}
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
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_SERIAL_TESTS}
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
        ${TEST_MPI_THREADS}
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
    FEATURES ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
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
    FEATURES ${TECA_HAS_MPI} ${TECA_HAS_NETCDF} ${TEST_MPI_THREADS}
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
    COMMAND test_binary_segmentation
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3.historical_r1i1p1_19500101-19501231\\.nc"
    prw 50 75 "prw_segmentation_50-75_%t%.%e%"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_serial
    COMMAND test_bayesian_ar_detect
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_%t%.nc" 1 0 -1 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_SERIAL_TESTS}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_threads
    EXEC_NAME test_bayesian_ar_detect
    SOURCES test_bayesian_ar_detect.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_bayesian_ar_detect
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_%t%.nc" -1 0 -1 1
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_bayesian_ar_detect
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_%t%.nc" -1 0 -1 1
    FEATURES ${TECA_HAS_MPI} ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_bayesian_ar_detect
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_%t%.nc" -1 0 -1 1
    FEATURES ${TECA_HAS_MPI} ${TECA_HAS_NETCDF} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_pass_through
    EXEC_NAME test_normalize_coordinates
    SOURCES test_normalize_coordinates.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_normalize_coordinates 90 45 7 0 360 -90 90 100 0
    "${TECA_DATA_ROOT}/test_normalize_coordinates.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_y
    COMMAND test_normalize_coordinates 90 45 7 0 360 90 -90 100 0
    "${TECA_DATA_ROOT}/test_normalize_coordinates.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_shift_x
    COMMAND test_normalize_coordinates 90 45 7 -180 180 -90 90 100 0
    "${TECA_DATA_ROOT}/test_normalize_coordinates_shift_x.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_shift_x_flip_y
    COMMAND test_normalize_coordinates 90 45 7 -180 180 90 -90 100 0
    "${TECA_DATA_ROOT}/test_normalize_coordinates_shift_x.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_pass_through_subset
    COMMAND test_normalize_coordinates 90 45 7 0 360 -90 90 100 0 40 190 -30 45 70 30
    "${TECA_DATA_ROOT}/test_normalize_coordinates_subset.bin"
    REQ_TECA_DATA)

teca_add_test(test_normalize_coordinates_flip_y_subset
    COMMAND test_normalize_coordinates 90 45 7 0 360 90 -90 100 0 40 190 -30 45 70 30
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
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_collective_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} test_cf_writer_collective 128 512 128 2
    "${TECA_DATA_ROOT}/test_cf_writer_collective_%t%.bin" 213
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_bad_type
    SOURCES test_cf_writer_bad_type.cpp
    LIBS teca_core teca_data teca_alg teca_io ${teca_test_link}
    COMMAND test_cf_writer_bad_type
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA
    WILL_FAIL)

teca_add_test(test_integrated_water_vapor_vv_mask
    SOURCES test_integrated_water_vapor.cpp
    EXEC_NAME test_integrated_water_vapor
    LIBS teca_core teca_data teca_alg teca_io ${teca_test_link}
    COMMAND test_integrated_water_vapor 0 0 0
    FEATURES ${TECA_HAS_NETCDF})

teca_add_test(test_integrated_water_vapor_elev_mask
    LIBS teca_core teca_data teca_alg teca_io ${teca_test_link}
    COMMAND test_integrated_water_vapor 1 0 0
    FEATURES ${TECA_HAS_NETCDF})

teca_add_test(test_integrated_vapor_transport_no_mask_fo
    SOURCES test_integrated_vapor_transport.cpp
    EXEC_NAME test_integrated_vapor_transport
    LIBS teca_core teca_data teca_alg teca_io ${teca_test_link}
    COMMAND test_integrated_vapor_transport 2 256 0 0 0 0
    FEATURES ${TECA_HAS_NETCDF})

teca_add_test(test_integrated_vapor_transport_no_mask_trap
    COMMAND test_integrated_vapor_transport 2 256 0 0 0 1
    FEATURES ${TECA_HAS_NETCDF})

teca_add_test(test_integrated_vapor_transport_vv_mask_trap
    COMMAND test_integrated_vapor_transport 2 256 1 0 0 1
    FEATURES ${TECA_HAS_NETCDF})

teca_add_test(test_integrated_vapor_transport_elev_mask_trap
    COMMAND test_integrated_vapor_transport 2 256 2 0 0 1
    FEATURES ${TECA_HAS_NETCDF})

teca_add_test(test_cf_time_axis_reader
    SOURCES test_cf_time_axis_reader.cpp
    LIBS teca_core teca_data teca_alg teca_io ${teca_test_link}
    COMMAND test_cf_time_axis_reader
        "${TECA_DATA_ROOT}/HighResMIP/TC_test/PSL/PSL.*\\.nc$" 1
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_time_axis_reader_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_time_axis_reader
        "${TECA_DATA_ROOT}/HighResMIP/TC_test/PSL/PSL.*\\.nc$" ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_valid_value_mask
    SOURCES test_valid_value_mask.cpp
    LIBS teca_core teca_data teca_alg teca_io ${teca_test_link}
    COMMAND test_valid_value_mask 0.25 1.0e20 "${TECA_DATA_ROOT}/test_valid_value_mask.nc"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_unpack_data
    SOURCES test_unpack_data.cpp
    LIBS teca_core teca_data teca_alg teca_io ${teca_test_link}
    COMMAND test_unpack_data "${TECA_DATA_ROOT}/test_unpack_data.nc"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_priority_queue
    SOURCES test_priority_queue.cpp
    LIBS teca_core teca_data ${teca_test_link}
    COMMAND test_priority_queue 57 1)

teca_add_test(test_bounds_to_extent
    SOURCES test_bounds_to_extent.cpp
    LIBS teca_core teca_data ${teca_test_link}
    COMMAND test_bounds_to_extent)

teca_add_test(test_elevation_mask
    SOURCES test_elevation_mask.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_elevation_mask cf
    "${TECA_DATA_ROOT}/ECMWF-IFS-HR-SST-present_himalaya\\.nc"
    "${TECA_DATA_ROOT}/GTOPO_DEM_025deg\\.nc"
    "${TECA_DATA_ROOT}/test_elevation_mask"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_interval_iterator_daily
    SOURCES test_interval_iterator.cpp
    EXEC_NAME test_interval_iterator
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_interval_iterator daily
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
    365
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_interval_iterator_monthly
    COMMAND test_interval_iterator monthly
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
    12
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_interval_iterator_seasonal
    COMMAND test_interval_iterator seasonal
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
    3
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_interval_iterator_yearly
    COMMAND test_interval_iterator yearly
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
    1
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_rename_variables
    SOURCES test_rename_variables.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_rename_variables "${TECA_DATA_ROOT}/test_rename_variables.nc"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_shape_file_mask
    SOURCES test_shape_file_mask.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_shape_file_mask ${TECA_DATA_ROOT}/GMBAMountainInventory_v1.2-World.shp
        mountain_mask 360 180 ${TECA_DATA_ROOT}/GMBAMountainInventory_v1.2-World.nc
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_SHAPELIB}
    REQ_TECA_DATA)

teca_add_test(test_cartesian_mesh_coordinate_transform
    SOURCES test_cartesian_mesh_coordinate_transform.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_cartesian_mesh_coordinate_transform
        "${TECA_DATA_ROOT}/test_cartesian_mesh_coordinate_transform.nc"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_time_axis_convolution_backward_constant
    EXEC_NAME test_time_axis_convolution
    SOURCES test_time_axis_convolution.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_time_axis_convolution
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
    "${TECA_DATA_ROOT}/test_time_axis_convolution_backward_constant"
    constant 3 backward 0 0 30 1 prw
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_time_axis_convolution_forward_constant
    COMMAND test_time_axis_convolution
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
    "${TECA_DATA_ROOT}/test_time_axis_convolution_forward_constant"
    constant 3 forward 0 0 30 1 prw
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_time_axis_convolution_centered_gaussian
    COMMAND test_time_axis_convolution
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
    "${TECA_DATA_ROOT}/test_time_axis_convolution_centered_gaussian"
    gaussian 5 centered 0 0 30 1 prw
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_time_axis_convolution_centered_gaussian_highpass
    COMMAND test_time_axis_convolution
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
    "${TECA_DATA_ROOT}/test_time_axis_convolution_centered_gaussian_highpass"
    gaussian 5 centered 1 0 30 1 prw
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_variant_array_cuda_access
    SOURCES test_variant_array_cuda_access.cpp
    LIBS teca_core ${teca_test_link}
    COMMAND test_variant_array_cuda_access
    FEATURES ${TECA_HAS_CUDA})

teca_add_test(test_variant_array_cpu_access
    SOURCES test_variant_array_cpu_access.cpp
    LIBS teca_core ${teca_test_link}
    COMMAND test_variant_array_cpu_access)

teca_add_test(test_array_collection_cuda_access
    SOURCES test_array_collection_cuda_access.cpp
    LIBS teca_core teca_data ${teca_test_link}
    COMMAND test_array_collection_cuda_access
    FEATURES ${TECA_HAS_CUDA})

teca_add_test(test_l2_norm_threads
    SOURCES test_l2_norm_threads.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_l2_norm_threads
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-[0-9][0-9]-10800\\.nc"
    U850 V850 . W850 test_l2_norm_cam5_%t%.nc
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_l2_norm
    SOURCES test_l2_norm.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_l2_norm "${TECA_DATA_ROOT}/test_l2_norm.bin"
    REQ_TECA_DATA)

teca_add_test(test_space_time_executive_size
    SOURCES test_space_time_executive.cpp
    EXEC_NAME test_space_time_executive
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_space_time_executive 256 256 7 512 8 size 128)

teca_add_test(test_space_time_executive_number
    COMMAND test_space_time_executive 256 256 7 512 8 number 2)

teca_add_test(test_space_time_executive_number_par
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_space_time_executive
    256 256 7 512 8 number 2
    FEATURES ${TECA_HAS_MPI})

teca_add_test(test_space_time_executive_size_par
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_space_time_executive
    256 256 7 512 8 size 128
    FEATURES ${TECA_HAS_MPI})

teca_add_test(test_spatial_executive_size
    SOURCES test_spatial_executive.cpp
    EXEC_NAME test_spatial_executive
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_spatial_executive 256 256 7 512 size 128)

teca_add_test(test_spatial_executive_number
    COMMAND test_spatial_executive 256 256 7 512 number 2)

teca_add_test(test_spatial_executive_number_par
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_spatial_executive
    256 256 7 512 number 2
    FEATURES ${TECA_HAS_MPI})

teca_add_test(test_spatial_executive_size_par
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_spatial_executive
    256 256 7 512 size 128
    FEATURES ${TECA_HAS_MPI})

teca_add_test(test_cf_writer_temporal_partition
    SOURCES test_cf_writer_partitions.cpp
    EXEC_NAME test_cf_writer_partitions
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_cf_writer_partitions 129 64 1 257 monthly temporal 0 0 -1
        ${TECA_DATA_ROOT} test_cf_writer_partitons
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_temporal_partition_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_writer_partitions 129 64 1 257
        monthly temporal 0 0 -1 ${TECA_DATA_ROOT} test_cf_writer_partitons
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_space_time_partition
    COMMAND test_cf_writer_partitions 129 64 1 257 monthly space_time 17 32 -1
        ${TECA_DATA_ROOT} test_cf_writer_partitons
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_space_time_partition_blocking_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_writer_partitions 129 64 1 257
        monthly space_time 17 32 -1 ${TECA_DATA_ROOT} test_cf_writer_partitons
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_space_time_partition_single_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_writer_partitions 129 64 1 257
        monthly space_time 257 32 -1 ${TECA_DATA_ROOT} test_cf_writer_partitons
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_spatial_partition
    COMMAND test_cf_writer_partitions 129 64 1 257 monthly spatial 17 32 -1
        ${TECA_DATA_ROOT} test_cf_writer_partitons
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_spatial_partition_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_writer_partitions 129 64 1 257
        monthly spatial 17 32 -1 ${TECA_DATA_ROOT} test_cf_writer_partitons
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_cf_writer_spatial_partition_mpi_cb
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_writer_partitions 129 64 1 257
        monthly spatial 17 0 1 ${TECA_DATA_ROOT} test_cf_writer_partitons
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_cf_reader_temporal_partition
    SOURCES test_cf_reader_partitions.cpp
    EXEC_NAME test_cf_reader_partitions
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_cf_reader_partitions "${TECA_DATA_ROOT}/test_cf_writer_partitons\\.nc$"
        temporal 0 0 0 U V R
    REQ_TECA_DATA)

teca_add_test(test_cf_reader_temporal_partition_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_reader_partitions
        "${TECA_DATA_ROOT}/test_cf_writer_partitons\\.nc$" temporal 0 0 0 U V R
    FEATURES ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_cf_reader_space_time_partition
    COMMAND test_cf_reader_partitions "${TECA_DATA_ROOT}/test_cf_writer_partitons\\.nc$"
        space_time 8 8 0 U V R
    REQ_TECA_DATA)

teca_add_test(test_cf_reader_space_time_partition_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_reader_partitions
        "${TECA_DATA_ROOT}/test_cf_writer_partitons\\.nc$" space_time 8 8 0 U V R
    FEATURES ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_cf_reader_spatial_partition
    COMMAND test_cf_reader_partitions "${TECA_DATA_ROOT}/test_cf_writer_partitons\\.nc$"
        spatial 8 8 0 U V R
    REQ_TECA_DATA)

teca_add_test(test_cf_reader_spatial_partition_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_cf_reader_partitions
        "${TECA_DATA_ROOT}/test_cf_writer_partitons\\.nc$" spatial 8 8 0 U V R
    FEATURES ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_temporal_index_select_nompi
    EXEC_NAME test_temporal_index_select
    SOURCES test_temporal_index_select.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_temporal_index_select
        "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
        "${TECA_DATA_ROOT}/test_temporal_index_select_prw_monthly" "prw" 0 2 181 363
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_temporal_index_select_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_temporal_index_select
        "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
        "${TECA_DATA_ROOT}/test_temporal_index_select_prw_monthly" "prw" 0 2 181 363
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_temporal_index_select_mpi_spatial_partition
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_temporal_index_select
        "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
        "${TECA_DATA_ROOT}/test_temporal_index_select_prw_monthly" "prw" 1 2 181 363
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_temporal_index_select_mpi_space_time_partition
    COMMAND ${MPIEXEC} -n ${TEST_CORES} test_temporal_index_select
        "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc$"
        "${TECA_DATA_ROOT}/test_temporal_index_select_prw_monthly" "prw" 2 2 181 363
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_table_join
    SOURCES test_table_join.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_table_join)

teca_add_test(test_ha4_connected_components
    SOURCES test_ha4_connected_components.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_ha4_connected_components 256 256 14 1 1024 0
    FEATURES ${TECA_HAS_CUDA})

teca_add_test(test_ha4_connected_components_periodic
    COMMAND test_ha4_connected_components 256 256 14 1 1024 1
    FEATURES ${TECA_HAS_CUDA})

teca_add_test(test_cpp_temporal_reduction_inmem
    SOURCES test_cpp_temporal_reduction_inmem.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_cpp_temporal_reduction_inmem
    360 180 5 24 1 ${TEST_CORES} ${TEST_CORES} monthly average 1 #TODO -- more than 1 step per req
    test_cpp_temporal_reduction_inmem_%t%.nc yearly ${TEST_CORES} 1 1 1)

teca_add_test(test_cpp_temporal_reduction_io
    SOURCES test_cpp_temporal_reduction_io.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_cpp_temporal_reduction_io
    ${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\.nc
    lon lat . time prw 0 -1 ${TEST_CORES} ${TEST_CORES} ${TEST_CORES} 4 1 0
    monthly average 1 # TODO -- more than 1 step per req.
    ./test_cpp_temporal_reduction_io_%t%.nc yearly
    ${TEST_CORES} ${TEST_CORES} ${TEST_CORES} 1 1)

teca_add_test(test_regional_moisture_flux
    SOURCES test_regional_moisture_flux.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_regional_moisture_flux 512 0.002)

teca_add_test(test_regional_surface_flux
    SOURCES test_regional_surface_flux.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_regional_surface_flux 512 0.009)

teca_add_test(test_regional_surface_area
    SOURCES test_regional_surface_area.cpp
    LIBS teca_core teca_data teca_io teca_alg ${teca_test_link}
    COMMAND test_regional_surface_area
         512 "${TECA_DATA_ROOT}/vc965bq8111" 17814000.0 3e3
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_SHAPELIB}
    REQ_TECA_DATA)
