# teca_add_app(name
#   SOURCES  -- optional, source files to comile
#   LIBS -- optional, libraries to link to the compiled test
#   FEATURES -- optional, boolean condition decribing feature dependencies
#   )
function (teca_add_app app_name)
    set(opt_args)
    set(val_args FEATURES)
    set(array_args SOURCES LIBS)
    cmake_parse_arguments(APP
        "${opt_args}" "${val_args}" "${array_args}" ${ARGN})
    if (APP_FEATURES)
        if (NOT APP_SOURCES)
            set(APP_SOURCES "${app_name}.cpp")
        endif()
        add_executable(${app_name} ${APP_SOURCES})
        if (APP_LIBS)
            target_link_libraries(${app_name}
                teca_core teca_data teca_io teca_alg
                ${APP_LIBS})
        endif()
        install(TARGETS ${app_name} RUNTIME DESTINATION ${BIN_PREFIX})
    endif()
endfunction()
