# teca_add_test(name
#   EXEC_NAME -- optional, name of the compiled test
#   SOURCES  -- optional, source files to comile
#   LIBS -- optional, libraries to link to the compiled test
#   COMMAND -- required, test command
#   FEATURES -- optional, boolean condition decribing feature dependencies
#   REQ_TECA_DATA -- flag whose presence indicates the test needs the data repo
#   WILL_FAIL -- flag whose presence indicates the test is expected to fail
#   )
function (teca_add_test T_NAME)
    set(opt_args REQ_TECA_DATA WILL_FAIL)
    set(val_args EXEC_NAME)
    set(array_args SOURCES LIBS COMMAND FEATURES)
    cmake_parse_arguments(T "${opt_args}" "${val_args}" "${array_args}" ${ARGN})
    set(TEST_ENABLED ON)
    if (NOT DEFINED T_FEATURES)
        set(TEST_ENABLED ON)
    else()
        foreach(feature ${T_FEATURES})
            if (NOT feature)
                set(TEST_ENABLED OFF)
            endif()
        endforeach()
    endif()
    if (TEST_ENABLED)
        if (T_SOURCES)
            set(EXEC_NAME ${T_NAME})
            if (T_EXEC_NAME)
                set(EXEC_NAME ${T_EXEC_NAME})
            endif()
            add_executable(${EXEC_NAME} ${T_SOURCES})
            if (T_LIBS)
                target_link_libraries(${EXEC_NAME} ${T_LIBS})
            endif()
        endif()
        if ((T_REQ_TECA_DATA AND TECA_HAS_DATA) OR NOT T_REQ_TECA_DATA)
            add_test(NAME ${T_NAME} COMMAND ${T_COMMAND}
                WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
            if (T_WILL_FAIL)
                set_tests_properties(${T_NAME}
                    PROPERTIES PASS_REGULAR_EXPRESSION "ERROR;FAIL")
            else()
                set_tests_properties(${T_NAME}
                    PROPERTIES FAIL_REGULAR_EXPRESSION "ERROR;FAIL")
            endif()
            if (TECA_ENABLE_PROFILER)
                set_property(TEST ${T_NAME} APPEND PROPERTY ENVIRONMENT
                    "PROFILER_ENABLE=3;PROFILER_LOG_FILE=${T_NAME}_time.csv;"
                    "MEMPROF_INTERVAL=0.5;MEMPROF_LOG_FILE=${T_NAME}_mem.csv")
            endif()
        endif()
    endif()
endfunction()
