set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_alg,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    )

teca_add_test(test_deeplabv3p_ar_detect_app_serial
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_deeplabv3p_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_deeplabv3p_ar_detect_app_mpi
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_deeplabv3p_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} ${MPIEXEC} 2
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_app_serial
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_bayesian_ar_detect_app_mpi
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_bayesian_ar_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} ${MPIEXEC} 2
    FEATURES ${TECA_HAS_NETCDF} ${TECA_HAS_MPI}
    REQ_TECA_DATA)

teca_add_test(test_tc_detect_app_serial
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_tc_detect_app_mpi
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_detect_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} ${MPIEXEC} 2
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_tc_trajectory_app
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_trajectory_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_app_serial
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_wind_radii_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    FEATURES ${TECA_HAS_NETCDF}
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_app_mpi
    LIBS teca_io teca_alg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_wind_radii_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} ${MPIEXEC} 2
    FEATURES ${TECA_HAS_NETCDF}
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
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT} ${MPIEXEC} 2
    REQ_TECA_DATA)

teca_add_test(test_tc_wind_radii_stats_app
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_tc_wind_radii_stats_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)

teca_add_test(test_event_filter_app
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_event_filter_app.sh
    ${CMAKE_BINARY_DIR}/${BIN_PREFIX} ${TECA_DATA_ROOT}
    REQ_TECA_DATA)
