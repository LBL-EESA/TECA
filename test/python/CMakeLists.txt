teca_add_test(py_test_cf_reader
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_reader.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    1 2 "py_test_cf_reader_%t%.vtk" U850 V850
    FEATURES ${TECA_HAS_NETCDF}
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
    COMMAND  ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_writer.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    0 -1 1 2 "py_test_cf_writer_%t%.nc" U850 V850
    FEATURES ${TECA_HAS_MPI} ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_cf_writer_mpi_threads
    COMMAND  ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_writer.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    0 -1 ${HALF_TEST_CORES} 1 "py_test_cf_writer_%t%.nc" U850 V850
    FEATURES ${TECA_HAS_MPI} ${TECA_HAS_NETCDF}
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
    "${TECA_DATA_ROOT}/py_test_programmable_reduce.bin" 0 -1 1 TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_programmable_reduce_thread
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_programmable_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    "${TECA_DATA_ROOT}/py_test_programmable_reduce.bin" 0 -1 4 TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_programmable_reduce_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_programmable_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    "${TECA_DATA_ROOT}/py_test_programmable_reduce.bin" 0 -1 1 TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_programmable_reduce_mpi_thread
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_programmable_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    "${TECA_DATA_ROOT}/py_test_programmable_reduce.bin" 0 -1 ${HALF_TEST_CORES}
    TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_python_reduce_serial
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_python_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    "${TECA_DATA_ROOT}/py_test_programmable_reduce.bin" 0 -1 1 TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_python_reduce_thread
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_python_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    "${TECA_DATA_ROOT}/py_test_programmable_reduce.bin" 0 -1 4 TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_python_reduce_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_python_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    "${TECA_DATA_ROOT}/py_test_programmable_reduce.bin" 0 -1 1 TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_python_reduce_mpi_thread
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_python_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    "${TECA_DATA_ROOT}/py_test_programmable_reduce.bin" 0 -1 ${HALF_TEST_CORES}
    TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_monthly_average_thread
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "${TECA_DATA_ROOT}/prw" 7 2 monthly average prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_monthly_average_mpi_thread
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "${TECA_DATA_ROOT}/prw" 7 2 monthly average prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_monthly_minimum_thread
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "${TECA_DATA_ROOT}/prw" 7 2 monthly minimum prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_monthly_minimum_mpi_thread
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "${TECA_DATA_ROOT}/prw" 7 2 monthly minimum prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_monthly_maximum_thread
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "${TECA_DATA_ROOT}/prw" 7 2 monthly maximum prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_temporal_monthly_maximum_mpi_thread
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_temporal_reduction.py
    "${TECA_DATA_ROOT}/prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\\.nc"
    "${TECA_DATA_ROOT}/prw" 7 2 monthly maximum prw
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_tc_candidates_serial
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_candidates.py
    "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin" 0 3 1
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS}
    REQ_TECA_DATA)

teca_add_test(py_test_tc_candidates_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE}
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
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_candidates.py
    "${TECA_DATA_ROOT}/test_tc_candidates_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_candidates_20.bin" 0 3 ${HALF_TEST_CORES}
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_UDUNITS} ${TECA_HAS_MPI}
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
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE}
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

teca_add_test(py_test_bayesian_ar_detect_serial
    COMMAND  ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect.py
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_py_%t%.vtk" 1 0 4
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_bayesian_ar_detect_threads
    COMMAND  ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect.py
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_py_%t%.vtk" -1 0 4
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_bayesian_ar_detect_mpi_threads
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect.py
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D_2017-05.*\\.nc$"
    "${TECA_DATA_ROOT}/test_bayesian_ar_detect.bin" IVT
    "bayesian_ar_detect_py_%t%.vtk"
    ${HALF_TEST_CORES} 0 4
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(py_test_deeplabv3p_ar_detect
    COMMAND  ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_deeplabv3p_ar_detect.py
    "${TECA_DATA_ROOT}/cascade_deeplab_IVT.pt"
    "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D.*\.nc$"
    "${TECA_DATA_ROOT}/test_deeplabv3p_ar_detect.bin" IVT 1
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_deeplabv3p_ar_detect_mpi
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_deeplabv3p_ar_detect.py
       "${TECA_DATA_ROOT}/cascade_deeplab_IVT.pt"
        "${TECA_DATA_ROOT}/ARTMIP_MERRA_2D.*\.nc$"
        "${TECA_DATA_ROOT}/test_deeplabv3p_ar_detect.bin" IVT 1
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_binary_stream
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_binary_stream.py
    "${TECA_DATA_ROOT}/py_test_binary_stream.bin"
    FEATURES ${TECA_HAS_MPI} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_nested_pipeline
    COMMAND ${MPIEXEC} -n ${HALF_TEST_CORES} ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_nested_pipeline.py 16 16 32 ${HALF_TEST_CORES}
    FEATURES ${TECA_HAS_MPI} ${MPI4Py_FOUND})

teca_add_test(py_test_cf_writer_collective_serial
    COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/test_cf_writer_collective.py" 128 512 128
    "${TECA_DATA_ROOT}/test_cf_writer_collective_%t%.bin" 213
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_cf_writer_collective_mpi
    COMMAND ${MPIEXEC} -n ${TEST_CORES} ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/test_cf_writer_collective.py" 128 512 128
    "${TECA_DATA_ROOT}/test_cf_writer_collective_%t%.bin" 213
    FEATURES ${TECA_HAS_NETCDF} ${MPI4Py_FOUND}
    REQ_TECA_DATA)

teca_add_test(py_test_config
    COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/test_config.py")
