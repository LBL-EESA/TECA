set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_alg,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    )

teca_add_app_test(test_bayesian_ar_detect_app_threads
    teca_bayesian_ar_detect
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1 2
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_app_test(test_bayesian_ar_detect_app_mpi
    teca_bayesian_ar_detect
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1 2
    ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_app_test(test_bayesian_ar_detect_app_mpi_threads
    teca_bayesian_ar_detect
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1 2
    ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_app_test(test_bayesian_ar_detect_app_mcf_threads
    teca_bayesian_ar_detect
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1 2
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_app_test(test_bayesian_ar_detect_app_mcf_mpi
    teca_bayesian_ar_detect
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1 2
    ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_app_test(test_bayesian_ar_detect_app_mcf_mpi_threads
    teca_bayesian_ar_detect
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1 2
    ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_app_test(test_bayesian_ar_detect_app_packed_data_mpi
    teca_bayesian_ar_detect
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app_packed_data.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1 2
    ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_app_test(test_integrated_water_vapor_app_threads
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_integrated_water_vapor_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1 2
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_app_test(test_integrated_water_vapor_app_mpi
    teca_integrated_water_vapor
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_integrated_water_vapor_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_app_test(test_integrated_water_vapor_app_mpi_threads
    teca_integrated_water_vapor
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_integrated_water_vapor_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_app_test(test_integrated_vapor_transport_app_threads
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_integrated_vapor_transport_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_app_test(test_integrated_vapor_transport_app_mpi
    teca_integrated_vapor_transport
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_integrated_vapor_transport_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_app_test(test_integrated_vapor_transport_app_mpi_threads
    teca_integrated_vapor_transport
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_integrated_vapor_transport_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_app_test(test_integrated_vapor_transport_app_packed_data
    teca_integrated_vapor_transport
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_integrated_vapor_transport_app_packed_data.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_app_test(test_tc_detect_app teca_tc_detect
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_app_test(test_tc_detect_app_mpi teca_tc_detect
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_app_test(test_tc_detect_app_mcf teca_tc_detect
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_app_test(test_tc_trajectory_app
    teca_tc_trajectory
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_trajectory_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)

teca_add_app_test(test_tc_wind_radii_app_serial
    teca_tc_wiind_radii
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_wind_radii_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_app_test(test_tc_wind_radii_app_mpi
    teca_tc_wiind_radii
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_wind_radii_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_app_test(test_tc_stats_app teca_tc_stats
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_stats_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)

teca_add_app_test(test_tc_trajectory_scalars_app_serial
    teca_tc_trajectory_scalars
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_trajectory_scalars_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)

teca_add_app_test(test_tc_trajectory_scalars_app_mpi
    teca_tc_trajectory_scalars
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_trajectory_scalars_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_MPI} ${MPI4Py_FOUND} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_app_test(test_tc_wind_radii_stats_app teca_tc_wind_radii_stats
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_wind_radii_stats_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)

teca_add_app_test(test_event_filter_app teca_event_filter
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_event_filter_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)

# test Python implementation of the temporal_reduction algorithm
teca_add_app_test(test_python_temporal_reduction_app_seasonal_average_py
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    seasonal average 7 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_seasonal_average_py_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    seasonal average 7 0 1 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_average_py
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly average 7 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_all_average_py
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19500131.nc" prw
    all average 7 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_average_py_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly average 7 0 1 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_py_minimum
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly minimum 7 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_minimum_py_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly minimum 7 0 1 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_maximum_py
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly maximum 7 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_maximum_py_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly maximum 7 0 1 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_daily_average_missing_values_py
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T daily average 1 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_daily_maximum_missing_values_py
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T daily maximum 1 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_daily_minimum_missing_values_py
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T daily minimum 1 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_mcf_daily_average_py
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "${CMAKE_BINARY_DIR}/${BIN_PREFIX}/../test/ECMWF-IFS-HR-SST-present.mcf" hus
    daily average 7 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_mcf_daily_average_py_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "${CMAKE_BINARY_DIR}/${BIN_PREFIX}/../test/ECMWF-IFS-HR-SST-present.mcf" hus
    daily average 7 0 1 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_spatial_partitioning_py
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    seasonal average 7 1 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_spatial_partitioning_py_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    seasonal average 7 1 1 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

# test C++ implementation of the temporal_reduction algorithm
teca_add_app_test(test_python_temporal_reduction_app_seasonal_average
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    seasonal average 7 0 0
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_seasonal_average_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    seasonal average 7 0 0 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_average
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly average 7 0 0
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_all_average
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19500131.nc" prw
    all average 7 0 0
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_average_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly average 7 0 0 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_minimum
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly minimum 7 0 0
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_minimum_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly minimum 7 0 0 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_maximum
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly maximum 7 0 0
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_maximum_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly maximum 7 0 0 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_daily_average_missing_values
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T daily average 1 0 0
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_daily_maximum_missing_values
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T daily maximum 1 0 0
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_daily_minimum_missing_values
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T daily minimum 1 0 0
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_mcf_daily_average
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "${CMAKE_BINARY_DIR}/${BIN_PREFIX}/../test/ECMWF-IFS-HR-SST-present.mcf" hus
    daily average 7 0 0
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_mcf_daily_average_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "${CMAKE_BINARY_DIR}/${BIN_PREFIX}/../test/ECMWF-IFS-HR-SST-present.mcf" hus
    daily average 7 0 0 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_spatial_partitioning
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    seasonal average 7 1 0
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_spatial_partitioning_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    seasonal average 7 1 0 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_seasonal_average
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    seasonal average 7 -1 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_seasonal_average_mpi_threads
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    seasonal average 7 -1 0 1 ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)


teca_add_app_test(test_cpp_temporal_reduction_app_monthly_average_serial
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly average 7 1 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_monthly_average
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly average 7 -1 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_monthly_average_mpi
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly average 7 1 0 1 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_monthly_average_mpi_threads
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly average 7 -1 0 1 ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_monthly_minimum
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly minimum 7 -1 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_monthly_minimum_mpi_threads
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly minimum 7 -1 0 1 ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_monthly_maximum
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly maximum 7 -1 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_monthly_maximum_mpi_threads
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly maximum 7 -1 0 1 ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_daily_average_missing_values
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T daily average 1 -1 0 2
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_daily_maximum_missing_values
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T daily maximum 1 -1 0 4
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_daily_minimum_missing_values
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T daily minimum 1 -1 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_n_steps_average_missing_values
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" T 8_steps average 1 -1 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_mcf_daily_average
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "${CMAKE_BINARY_DIR}/${BIN_PREFIX}/../test/ECMWF-IFS-HR-SST-present.mcf" hus
    daily average 7 -1 0 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_mcf_daily_average_mpi_threads
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app_mcf.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "${CMAKE_BINARY_DIR}/${BIN_PREFIX}/../test/ECMWF-IFS-HR-SST-present.mcf" hus
    daily average 7 -1 0 1 ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_spatial_partitioning
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    seasonal average 7 -1 1 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_cpp_temporal_reduction_app_spatial_partitioning_mpi_threads
    teca_cpp_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    seasonal average 7 -1 1 1 ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_percentile
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly 50th_percentile 7 0 0
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_python_temporal_reduction_app_monthly_percentile_mpi
    teca_temporal_reduction
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_python_temporal_reduction_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc" prw
    monthly 50th_percentile 7 0 0 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_cartesian_mesh_diff_app_pass teca_cartesian_mesh_diff
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cartesian_mesh_diff_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    ${TECA_DATA_ROOT}/test_cartesian_mesh_regrid.bin
    ${TECA_DATA_ROOT}/test_cartesian_mesh_regrid.bin LSMASK
    REQ_TECA_DATA)

teca_add_app_test(test_cartesian_mesh_diff_app_fail teca_cartesian_mesh_diff
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cartesian_mesh_diff_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    ${TECA_DATA_ROOT}/test_cf_writer_collective_000213.bin
    ${TECA_DATA_ROOT}/test_cartesian_mesh_regrid.bin LSMASK
    REQ_TECA_DATA
    WILL_FAIL)

teca_add_app_test(test_table_diff_app_pass teca_table_diff
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_table_diff_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    ${TECA_DATA_ROOT}/cam5_1_amip_run2_tracks_2005_09.bin
    ${TECA_DATA_ROOT}/cam5_1_amip_run2_tracks_2005_09.bin
    REQ_TECA_DATA)

teca_add_app_test(test_table_diff_app_fail teca_table_diff
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_table_diff_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    ${TECA_DATA_ROOT}/test_tc_stats_app_output_class_table_ref.bin
    ${TECA_DATA_ROOT}/cam5_1_amip_run2_tracks_2005_09.bin
    REQ_TECA_DATA
    WILL_FAIL)

teca_add_app_test(test_convert_table_app_csv_bin teca_convert_table
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_convert_table_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "cam5-025deg-all-hist-est1-v3-r1-tracks_size.csv"
    "cam5-025deg-all-hist-est1-v3-r1-tracks_size.bin"
    REQ_TECA_DATA)

teca_add_app_test(test_convert_table_app_bin_csv teca_convert_table
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_convert_table_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "cam5-025deg-all-hist-est1-v3-r1-tracks_size.bin"
    "cam5-025deg-all-hist-est1-v3-r1-tracks_size.csv"
    REQ_TECA_DATA)

teca_add_app_test(test_convert_table_app_bin_netcdf teca_convert_table
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_convert_table_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "test_descriptive_statistics.bin"
    "test_descriptive_statistics.nc"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_app_test(test_metadata_probe_app_cf teca_metadata_probe
    COMMAND ${CMAKE_BINARY_DIR}/${BIN_PREFIX}/teca_metadata_probe
    --input_regex "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[0-9]\\.nc"
    REQ_TECA_DATA)

teca_add_app_test(test_metadata_probe_app_mcf teca_metadata_probe
    COMMAND ${CMAKE_BINARY_DIR}/${BIN_PREFIX}/teca_metadata_probe
    --input_file ${CMAKE_CURRENT_BINARY_DIR}/../ECMWF-IFS-HR-SST-present.mcf
    --z_axis plev
    REQ_TECA_DATA)

teca_add_app_test(test_cf_restripe_app teca_cf_restripe
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_restripe_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    REQ_TECA_DATA)

teca_add_app_test(test_cf_restripe_app_mpi teca_cf_restripe
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_restripe_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_app_test(test_potential_intensity_app teca_potential_intensity
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_potential_intensity_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} 0
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_TCPYPI}
    REQ_TECA_DATA)

teca_add_app_test(test_potential_intensity_app_mpi teca_potential_intensity
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_potential_intensity_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} 0 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI} ${TECA_HAS_TCPYPI}
    REQ_TECA_DATA)

teca_add_app_test(test_potential_intensity_app_mpi_spatial_par teca_potential_intensity
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_potential_intensity_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} 1 ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI} ${TECA_HAS_TCPYPI}
    REQ_TECA_DATA)

teca_add_app_test(test_lapse_rate_app teca_lapse_rate
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_lapse_rate_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX}
    REQ_TECA_DATA)

teca_add_app_test(test_spectral_filter_app
    teca_spectral_filter
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_spectral_filter_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    prw high_pass 0 5 days
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_app_test(test_spectral_filter_app_mpi
    teca_spectral_filter
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_spectral_filter_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    "prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    prw high_pass 0 5 days ${MPIEXEC} ${TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_app_test(test_regional_moisture_flux_app_threads
    teca_regional_moisture_flux
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_regional_moisture_flux_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_app_test(test_regional_moisture_flux_app_mpi_threads
    teca_regional_moisture_flux
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_regional_moisture_flux_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} -1
    ${MPIEXEC} ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

set(app_names
    teca_bayesian_ar_detect
    teca_cf_restripe
    teca_convert_table
    teca_deeplab_ar_detect
    teca_event_filter
    teca_integrated_vapor_transport
    teca_lapse_rate
    teca_metadata_probe
    teca_potential_intensity
    teca_profile_explorer
    teca_spectral_filter
    teca_tc_detect
    teca_tc_stats
    teca_tc_trajectory
    teca_tc_trajectory_scalars
    teca_tc_wind_radii
    teca_tc_wind_radii_stats
    teca_temporal_reduction
    teca_cpp_temporal_reduction)

foreach (app_name ${app_names})

    if (TARGET ${app_name})

        teca_add_app_test(${app_name}_help ${app_name}
            COMMAND ${CMAKE_BINARY_DIR}/${BIN_PREFIX}/${app_name} --help)

        get_target_property(app_type ${app_name} APP_TYPE)
        if ("${app_type}" STREQUAL "C++")

            teca_add_app_test(${app_name}_full_help ${app_name}
                COMMAND ${CMAKE_BINARY_DIR}/${BIN_PREFIX}/${app_name} --full_help)

            teca_add_app_test(${app_name}_advanced_help ${app_name}
                COMMAND ${CMAKE_BINARY_DIR}/${BIN_PREFIX}/${app_name} --advanced_help)

            teca_add_app_test(${app_name}_no_args ${app_name}
                COMMAND ${CMAKE_BINARY_DIR}/${BIN_PREFIX}/${app_name}
                WILL_FAIL)
        endif()

    endif()
endforeach()
