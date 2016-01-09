if (TECA_DATA_ROOT)
    add_test(NAME py_test_cf_reader
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_cf_reader.py
        "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-0[12]-10800\\.nc"
        1 2 "py_test_cf_reader_%t%.vtk" U850 V850
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    add_test(NAME py_test_connected_components
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_connected_components.py
        "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-01-10800\\.nc"
        "" U850 V850 15 0 0 "py_test_connected_components_%t%.vtk"
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    add_test(NAME py_test_programmable_algorithm
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_programmable_algorithm.py
        "${TECA_DATA_ROOT}/cam5_1_amip_run2\\.cam2\\.h2\\.1991-10-01-10800\\.nc"
        U850 V850 0 3 "py_test_programmable_algorithm_%t%.vtk"
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

add_test(NAME py_test_array_collection
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_array_collection.py
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_test(NAME py_test_metadata
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_metadata.py
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_test(NAME py_test_variant_array
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/test_variant_array.py
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
