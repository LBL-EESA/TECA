teca_add_test(py_test_cf_reader
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_reader.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    1 2 "py_test_cf_reader_%t%.vtk" U850 V850
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

# TODO -- camke_parse_arguments eats ""
if (TECA_DATA_ROOT AND TECA_HAS_NETCDF)
    add_test(NAME py_test_connected_components
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_connected_components.py
        "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-01-10800\\.nc"
        "" U850 V850 15 0 0 "py_test_connected_components_%t%.vtk"
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

teca_add_test(py_test_derived_quantity
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_derived_quantity.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    U850 V850 0 0 "py_test_derived_quantity_%t%.vtk"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_programmable_algorithm
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_programmable_algorithm.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    U850 V850 0 0 "py_test_programmable_algorithm_%t%.vtk"
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_vector_ops
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_vector_ops.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-01-10800\\.nc"
    0 0 py_test_vector_ops_%t%.vtk U850 V850
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_array_collection
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_array_collection.py)

teca_add_test(py_test_metadata
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_metadata.py)

teca_add_test(py_test_variant_array
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_variant_array.py)

teca_add_test(py_test_database
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_database.py)

teca_add_test(py_test_programmable_map_reduce_serial
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_programmable_map_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    py_test_programmable_map_reduce.bin 0 -1 1 TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_programmable_map_reduce_thread
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_programmable_map_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    py_test_programmable_map_reduce.bin 0 -1 4 TMQ T200 T500
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_programmable_map_reduce_mpi
    COMMAND mpiexec -np 4 python ${CMAKE_CURRENT_SOURCE_DIR}/test_programmable_map_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    py_test_programmable_map_reduce.bin 0 -1 1 TMQ T200 T500
    FEATURES (${TECA_HAS_NETCDF} AND ${TECA_HAS_MPI} AND ${MPI4PY_FOUND})
    REQ_TECA_DATA)

teca_add_test(py_test_programmable_map_reduce_mpi_thread
    COMMAND mpiexec -np 2 python ${CMAKE_CURRENT_SOURCE_DIR}/test_programmable_map_reduce.py
    "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
    py_test_programmable_map_reduce.bin 0 -1 2 TMQ T200 T500
    FEATURES (${TECA_HAS_NETCDF} AND ${TECA_HAS_MPI} AND ${MPI4PY_FOUND})
    REQ_TECA_DATA)


teca_add_test(py_test_tc_detect_serial
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect.py
    "${TECA_DATA_ROOT}/test_tc_detect_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_detect_20.bin" 0 3 1
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(py_test_tc_detect_mpi
    COMMAND mpiexec -np 4 python ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect.py
    "${TECA_DATA_ROOT}/test_tc_detect_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_detect_20.bin" 0 3 1
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES (${TECA_HAS_NETCDF} AND ${TECA_HAS_MPI})
    REQ_TECA_DATA)

teca_add_test(py_test_tc_detect_threads
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect.py
    "${TECA_DATA_ROOT}/test_tc_detect_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_detect_20.bin" 0 3 4
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES (${TECA_HAS_NETCDF} AND ${TECA_HAS_MPI})
    REQ_TECA_DATA)

teca_add_test(py_test_tc_detect_mpi_threads
    COMMAND mpiexec -np 2 python ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect.py
    "${TECA_DATA_ROOT}/test_tc_detect_1990_07_0[12]\\.nc"
    "${TECA_DATA_ROOT}/test_tc_detect_20.bin" 0 3 2
    U850 V850 UBOT VBOT PSL T500 T200 Z1000 Z200 -20 20
    FEATURES (${TECA_HAS_NETCDF} AND ${TECA_HAS_MPI})
    REQ_TECA_DATA)
