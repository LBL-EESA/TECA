set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_alg,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    )

teca_add_test(test_deeplabv3p_ar_detect_app_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_deeplabv3p_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_PYTORCH}
    REQ_TECA_DATA)

teca_add_test(test_deeplabv3p_ar_detect_app_mpi
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_deeplabv3p_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_PYTORCH}
        ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_app_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_app_mpi
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_app_mcf_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_app_mcf_mpi
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_integrated_vapor_transport_app_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_integrated_vapor_transport_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_integrated_vapor_transport_app_mpi
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_integrated_vapor_transport_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_tc_detect_app
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_tc_detect_app_mpi
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_tc_detect_app_mcf
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_tc_trajectory_app
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_trajectory_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_app_serial
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_wind_radii_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_app_mpi
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_wind_radii_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_tc_stats_app
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_stats_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)

teca_add_test(test_tc_trajectory_scalars_app_serial
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_trajectory_scalars_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)

teca_add_test(test_tc_trajectory_scalars_app_mpi
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_trajectory_scalars_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_stats_app
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_wind_radii_stats_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)

teca_add_test(test_event_filter_app
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_event_filter_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)

teca_add_test(test_temporal_reduction_app_monthly_average_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly average 7
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(test_temporal_reduction_app_monthly_average_mpi_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly average 7 ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(test_temporal_reduction_app_monthly_minimum_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly minimum 7
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(test_temporal_reduction_app_monthly_minimum_mpi_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly minimum 7 ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(test_temporal_reduction_app_monthly_maximum_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly maximum 7
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(test_temporal_reduction_app_monthly_maximum_mpi_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly maximum 7 ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(test_temporal_reduction_app_daily_average_missing_values
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T daily average 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(test_temporal_reduction_app_daily_maximum_missing_values
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T daily maximum 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(test_temporal_reduction_app_daily_minimum_missing_values
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T daily minimum 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(test_cartesian_mesh_diff_app_pass
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cartesian_mesh_diff_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    ${TECA_DATA_ROOT}/test_cartesian_mesh_regrid.bin
    ${TECA_DATA_ROOT}/test_cartesian_mesh_regrid.bin LSMASK
    REQ_TECA_DATA)

teca_add_test(test_cartesian_mesh_diff_app_fail
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cartesian_mesh_diff_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    ${TECA_DATA_ROOT}/test_cf_writer_collective_000213.bin
    ${TECA_DATA_ROOT}/test_cartesian_mesh_regrid.bin LSMASK
    REQ_TECA_DATA
    WILL_FAIL)

teca_add_test(test_table_diff_app_pass
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_table_diff_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    ${TECA_DATA_ROOT}/cam5_1_amip_run2_tracks_2005_09.bin
    ${TECA_DATA_ROOT}/cam5_1_amip_run2_tracks_2005_09.bin
    REQ_TECA_DATA)

teca_add_test(test_table_diff_app_fail
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_table_diff_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    ${TECA_DATA_ROOT}/test_tc_stats_app_output_class_table_ref.bin
    ${TECA_DATA_ROOT}/cam5_1_amip_run2_tracks_2005_09.bin
    REQ_TECA_DATA
    WILL_FAIL)

teca_add_test(test_convert_table_app_csv_bin
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_convert_table_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "cam5-025deg-all-hist-est1-v3-r1-tracks_size.csv"
    "cam5-025deg-all-hist-est1-v3-r1-tracks_size.bin"
    REQ_TECA_DATA)

teca_add_test(test_convert_table_app_bin_csv
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_convert_table_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "cam5-025deg-all-hist-est1-v3-r1-tracks_size.bin"
    "cam5-025deg-all-hist-est1-v3-r1-tracks_size.csv"
    REQ_TECA_DATA)

teca_add_test(test_convert_table_app_bin_netcdf
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_convert_table_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "test_descriptive_statistics.bin"
    "test_descriptive_statistics.nc"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_metadata_probe_app_cf
    COMMAND ${CMAKE_BINARY_DIR}/${BIN_PREFIX}/teca_metadata_probe
    --input_regex "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[0-9]\\.nc"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_metadata_probe_app_mcf
    COMMAND ${CMAKE_BINARY_DIR}/${BIN_PREFIX}/teca_metadata_probe
    --input_file ${CMAKE_CURRENT_BINARY_DIR}/../ECMWF-IFS-HR-SST-present.mcf
    --z_axis plev
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_restripe_app
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_restripe_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_cf_restripe_app_mpi
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_restripe_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)
