# teca_add_test(name
#   SOURCES file_1 file_2 ...
#   LIBS lib_1 lib_2 ...
#   COMMAND cmd arg_1 arg_2 ...
#   FEATURES -- optional, boolean condition decribing featuire dependencies
#   REQ_TECA_DATA -- flag whose presence indicates the test needs the data repo
#   )
function (teca_add_test T_NAME)
    set(opt_args REQ_TECA_DATA)
    set(val_args)
    set(array_args SOURCES LIBS COMMAND FEATURES)
    cmake_parse_arguments(T
        "${opt_args}" "${val_args}" "${array_args}" ${ARGN})
    set(TEST_ENABLED OFF)
    if (${T_FEATURES})
        set(TEST_ENABLED ON)
    endif()
    if (NOT T_FEATURES)
        set(TEST_ENABLED ON)
    endif()
    if (TEST_ENABLED)
        if (T_SOURCES)
            add_executable(${T_NAME} ${T_SOURCES})
        endif()
        if (T_LIBS)
            target_link_libraries(${T_NAME} ${T_LIBS})
        endif()
        if ((REQ_TECA_DATA AND TECA_DATA_ROOT) OR NOT REQ_TECA_DATA)
            add_test(NAME ${T_NAME} COMMAND ${T_COMMAND}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
            set_tests_properties(${T_NAME}
                PROPERTIES FAIL_REGULAR_EXPRESSION "ERROR;FAIL")
        endif()
    endif()
endfunction()
