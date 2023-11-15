teca_add_test(py_test_cf_reader
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_reader.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    1 2 "py_test_cf_reader_%t%.vtk" U850 V850
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_multi_cf_reader_config_serial
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_cf_reader.py
    -g r_0 -t r_0 -o py_test_multi_cf_reader_config_serial_%t%.nc
    -f ${CMAKE_CURRENT_BINARY_DIR}/../ECMWF-IFS-HR-SST-present.mcf
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_multi_cf_reader_config_mpi
    COMMAND  ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_cf_reader.py
    -g r_0 -t r_0 -o py_test_multi_cf_reader_config_mpi_%t%.nc
    -f ${CMAKE_CURRENT_BINARY_DIR}/../ECMWF-IFS-HR-SST-present.mcf
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(py_test_multi_cf_reader_cmdln_serial
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_cf_reader.py
    -g r_0 -t r_0 -o py_test_multi_cf_reader_cmdln_serial_%t%.nc
    "(" "${TECA_DATA_ROOT}/HighResMIP/ECMWF-IFS-HR-SST-present/hus/hus_6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_1950-.*\\.nc" hus ")"
    "(" "${TECA_DATA_ROOT}/HighResMIP/ECMWF-IFS-HR-SST-present/ua/ua_6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_1950-.*\\.nc" ua ")"
    "(" "${TECA_DATA_ROOT}/HighResMIP/ECMWF-IFS-HR-SST-present/va/va_6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_1950-.*\\.nc" va ")"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_multi_cf_reader_cmdln_mpi
    COMMAND  ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_cf_reader.py
    -g r_0 -t r_0 -o py_test_multi_cf_reader_cmdln_mpi_%t%.nc
    "(" "${TECA_DATA_ROOT}/HighResMIP/ECMWF-IFS-HR-SST-present/hus/hus_6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_1950-.*\\.nc" hus ")"
    "(" "${TECA_DATA_ROOT}/HighResMIP/ECMWF-IFS-HR-SST-present/ua/ua_6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_1950-.*\\.nc" ua ")"
    "(" "${TECA_DATA_ROOT}/HighResMIP/ECMWF-IFS-HR-SST-present/va/va_6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_1950-.*\\.nc" va ")"
    FEATURES ${TECA_HAS_NETCDF_MPI} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(py_test_cf_writer_serial
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_writer.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    0 -1 1 2 "py_test_cf_writer_%t%.nc" U850 V850
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_cf_writer_threads
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_writer.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    0 -1 ${TEST_CORES} 2 "py_test_cf_writer_%t%.nc" U850 V850
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_cf_writer_mpi
    COMMAND  ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_writer.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    0 -1 1 2 "py_test_cf_writer_%t%.nc" U850 V850
    FEATURES ${TECA_HAS_MPI} ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_cf_writer_mpi_threads
    COMMAND  ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_writer.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    0 -1 2 1 "py_test_cf_writer_%t%.nc" U850 V850
    FEATURES ${TECA_HAS_MPI} ${TECA_HAS_NETCDF} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

# TODO -- camke_parse_arguments eats ""
if (TECA_DATA_ROOT AND TECA_HAS_NETCDF)
    add_test(NAME py_test_connected_components
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_connected_components.py
        "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-01-10800\\.nc"
        "" U850 V850 15 0 0 "py_test_connected_components_%t%.vtk"
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

teca_add_test(py_test_component_area_filter
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_component_area_filter.py
    720 360 8 8 10000000 0 "py_test_component_area_filter_%t%.%e%")

teca_add_test(py_test_derived_quantity
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_derived_quantity.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    U850 V850 0 0 "py_test_derived_quantity_%t%.vtk"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_latitude_damper
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_latitude_damper.py
    361 181 10 "py_test_latitude_damper_%t%.%e%")

teca_add_test(py_test_programmable_algorithm
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_programmable_algorithm.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    U850 V850 0 0 "py_test_programmable_algorithm_%t%.vtk"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_python_algorithm
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_python_algorithm.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    U850 V850 0 0 "py_test_programmable_algorithm_%t%.vtk"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_vector_ops
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_vector_ops.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-01-10800\\.nc"
    0 0 py_test_vector_ops_%t%.vtk U850 V850
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_array_collection
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_array_collection.py)

teca_add_test(py_test_metadata
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_metadata.py)

teca_add_test(py_test_variant_array
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_variant_array.py)

teca_add_test(py_test_table
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_table.py
    "${TECA_DATA_ROOT}/py_test_table.bin"
    REQ_TECA_DATA)

teca_add_test(py_test_database
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_database.py)

teca_add_test(py_test_programmable_reduce_serial
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_programmable_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    "${TECA_DATA_ROOT}/py_test_programmable_reduce.bin" 0 -1 TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_programmable_reduce_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_programmable_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    "${TECA_DATA_ROOT}/py_test_programmable_reduce.bin" 0 -1 TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_python_reduce_serial
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_python_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    "${TECA_DATA_ROOT}/py_test_programmable_reduce.bin" 0 -1 TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_python_reduce_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_python_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    "${TECA_DATA_ROOT}/py_test_programmable_reduce.bin" 0 -1 TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_seasonal_average
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/test_temporal_reduction_prw"  seasonal average 0 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_seasonal_average_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/test_temporal_reduction_prw"  seasonal average 0 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI}
        ${MPI4Py_FOUND} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_monthly_average
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/test_temporal_reduction_prw"  monthly average 0 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_monthly_average_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/test_temporal_reduction_prw"  monthly average 0 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI}
        ${MPI4Py_FOUND} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_monthly_minimum
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/test_temporal_reduction_prw"  monthly minimum 0 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_monthly_minimum_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/test_temporal_reduction_prw"  monthly minimum 0 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI}
        ${MPI4Py_FOUND} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_monthly_maximum
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/prw"  monthly maximum 0 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_monthly_maximum_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/test_temporal_reduction_prw"  monthly maximum 0 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI}
        ${MPI4Py_FOUND} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_daily_average
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" "."
    "${TECA_DATA_ROOT}/MERRA2_100_inst3_3d_asm_Np_crop_ignore_missing_T"  daily average 0 T H
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_daily_average_missing_values
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" "."
    "${TECA_DATA_ROOT}/MERRA2_100_inst3_3d_asm_Np_crop_T"  daily average 1 T H
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_daily_maximum
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" "."
    "${TECA_DATA_ROOT}/MERRA2_100_inst3_3d_asm_Np_crop_ignore_missing_T"  daily maximum 0 T H
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_daily_maximum_missing_values
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" "."
    "${TECA_DATA_ROOT}/MERRA2_100_inst3_3d_asm_Np_crop_T"  daily maximum 1 T H
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_daily_minimum
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" "."
    "${TECA_DATA_ROOT}/MERRA2_100_inst3_3d_asm_Np_crop_ignore_missing_T"  daily minimum 0 T H
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_daily_minimum_missing_values
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/MERRA2_100_inst3_3d_asm_Np_crop_1980012[89]\\.nc" "."
    "${TECA_DATA_ROOT}/MERRA2_100_inst3_3d_asm_Np_crop_T"  daily minimum 1 T H
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_custom_temporal_reduction
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_custom_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/test_custom_temporal_reduction_prw"  0 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_monthly_percentile
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_percentile.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/test_temporal_reduction_prw" monthly 50 0 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_monthly_percentile_vv
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_percentile.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/test_temporal_reduction_prw" monthly 50 1 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_monthly_percentile_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_percentile.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/test_temporal_reduction_prw" monthly 50 0 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_monthly_percentile_vv_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_percentile.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "." "${TECA_DATA_ROOT}/test_temporal_reduction_prw" monthly 50 1 prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(py_test_tc_candidates_serial
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_candidates.py
    "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin" 0 3 1
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_SERIAL_TESTS}
    REQ_TECA_DATA)

teca_add_test(py_test_tc_candidates_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_candidates.py
    "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin" 0 3 1
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(py_test_tc_candidates_threads
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_candidates.py
    "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin" 0 3 ${TEST_CORES}
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(py_test_tc_candidates_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_candidates.py
    "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin" 0 3 2
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_test(py_test_event_filter
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_event_filter.py
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin"
    "${TECA_DATA_ROOT}/test_event_filter.bin"
    REQ_TECA_DATA)

teca_add_test(py_test_tc_stats
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_stats.py
    "${TECA_DATA_ROOT}/cam5-025deg-all-hist-est1-v3-r1_class_table.bin"
    "${TECA_DATA_ROOT}/cam5-025deg-all-hist-est1-v3-r1_tc_stats.bin"
    FEATURES ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_tc_activity
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_activity.py
    "${TECA_DATA_ROOT}/cam5-025deg-all-hist-est1-v3-r1_class_table.bin"
    "${TECA_DATA_ROOT}/cam5-025deg-all-hist-est1-v3-r1_tc_activity.bin"
    FEATURES ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_tc_trajectory_scalars_serial
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_trajectory_scalars.py
    "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800_median_in_cat_wr.bin"
    "${TECA_DATA_ROOT}/earthmap4k.png"
    "${TECA_DATA_ROOT}/py_test_tc_trajectory_scalars.bin" 0 -1
    FEATURES ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_tc_trajectory_scalars_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_trajectory_scalars.py
    "${TECA_DATA_ROOT}/tracks_1990s_3hr_mdd_4800_median_in_cat_wr.bin"
    "${TECA_DATA_ROOT}/earthmap4k.png"
    "${TECA_DATA_ROOT}/py_test_tc_trajectory_scalars.bin" 0 -1
    FEATURES ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_tc_wind_radii_stats
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_wind_radii_stats.py
    "${TECA_DATA_ROOT}/cam5-025deg-all-hist-est1-v3-r1-tracks_size.bin"
    FEATURES ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_bayesian_ar_detect
    COMMAND  ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect.py
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_py_%t%.vtk" -1 0 -1
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_bayesian_ar_detect_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect.py
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_py_%t%.vtk" -1 0 -1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(py_test_bayesian_ar_detect_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect.py
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_py_%t%.vtk" -1 0 -1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI} ${TEST_MPI_THREADS}
    REQ_TECA_DATA)

teca_add_test(py_test_binary_stream
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_binary_stream.py
    "${TECA_DATA_ROOT}/py_test_binary_stream.bin"
    FEATURES ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_nested_pipeline_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_nested_pipeline.py 16 16 32 -1
    FEATURES ${TECA_HAS_MPI} ${MPI4Py_FOUND})

teca_add_test(py_test_nested_pipeline_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_nested_pipeline.py 16 16 32 -1
    FEATURES ${TECA_HAS_MPI} ${MPI4Py_FOUND} ${TEST_MPI_THREADS})

teca_add_test(py_test_cf_writer_collective_serial
    COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/test_cf_writer_collective.py" 128 512 128
    "${TECA_DATA_ROOT}/test_cf_writer_collective_%t%.bin" 213
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_cf_writer_collective_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    "${CMAKE_CURRENT_SOURCE_DIR}/test_cf_writer_collective.py" 128 512 128
    "${TECA_DATA_ROOT}/test_cf_writer_collective_%t%.bin" 213
    FEATURES ${TECA_HAS_NETCDF} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_config
    COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/test_config.py")

teca_add_test(py_test_thread_parameters_serial
    COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/test_thread_parameters.py"
    FEATURES ${MPI4Py_FOUND})

teca_add_test(py_test_thread_parameters_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    "${CMAKE_CURRENT_SOURCE_DIR}/test_thread_parameters_mpi.py"
    FEATURES ${MPI4Py_FOUND})

teca_add_test(py_test_thread_parameters_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE} -m mpi4py
    "${CMAKE_CURRENT_SOURCE_DIR}/test_thread_parameters_mpi.py"
    FEATURES ${MPI4Py_FOUND} ${TEST_MPI_THREADS})

teca_add_test(py_test_calendaring
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_calendaring.py)

teca_add_test(py_test_table_from_stream_csv
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_table_from_stream.py
        "${TECA_DATA_ROOT}/cam5-025deg-all-hist-est1-v3-r1-tracks_size.csv"
    REQ_TECA_DATA)

teca_add_test(py_test_table_from_stream_bin
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_table_from_stream.py
        "${TECA_DATA_ROOT}/cam5-025deg-all-hist-est1-v3-r1-tracks_size.bin"
    REQ_TECA_DATA)

teca_add_test(py_test_apply_binary_mask
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_apply_binary_mask.py
        128 128 mask_ 1e-5)

teca_add_test(py_test_variant_array_cupy_cpu
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_variant_array_cupy_cpu.py
    FEATURES ${TECA_HAS_CUDA} ${TECA_HAS_CUPY})

teca_add_test(py_test_variant_array_cupy_cuda
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_variant_array_cupy_cuda.py
    FEATURES ${TECA_HAS_CUDA} ${TECA_HAS_CUPY})

teca_add_test(py_test_variant_array_numpy_cpu
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_variant_array_numpy_cpu.py)

teca_add_test(py_test_variant_array_numpy_cuda
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_variant_array_numpy_cuda.py
    FEATURES ${TECA_HAS_CUDA})

teca_add_test(py_test_cartesian_mesh_multi_dim
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cartesian_mesh_multi_dim.py 32 32 10
    test_cartesian_mesh_multi_dim.nc 1)

teca_add_test(py_test_lapse_rate
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_lapse_rate.py
    32 32 10 1 "py_test_lapse_rate_%t%.%e%")

teca_add_test(py_test_spectral_filter_low_pass_analytic
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_spectral_filter.py
    90 45 2 1000 364 35 15 50 365 1 100 days low_pass 0 analytic
    "${TECA_DATA_ROOT}/test_spectral_filter_low_pass" 1)

teca_add_test(py_test_spectral_filter_high_pass_analytic
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_spectral_filter.py
    90 45 2 1000 364 35 15 50 365 1 100 days high_pass 0 analytic
    "${TECA_DATA_ROOT}/test_spectral_filter_high_pass" 1)

teca_add_test(py_normalize_coordinates_z_axis_units
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_normalize_coords_z_axis_units.py ${TECA_DATA_ROOT}
    REQ_TECA_DATA)
