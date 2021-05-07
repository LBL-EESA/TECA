# teca_add_app(name
#   SOURCES  -- optional, source files to comile
#   LIBS -- optional, libraries to link to the compiled test
#   FEATURES -- optional, boolean condition decribing feature dependencies
#   )
function (teca_add_app app_name)
    set(opt_args)
    set(val_args)
    set(array_args SOURCES LIBS FEATURES)
    cmake_parse_arguments(APP
        "${opt_args}" "${val_args}" "${array_args}" ${ARGN})
    set(APP_ENABLED ON)
    if (DEFINED APP_FEATURES)
        foreach(feature ${APP_FEATURES})
            if (NOT feature)
                set(APP_ENABLED OFF)
            endif()
        endforeach()
    endif()
    if (APP_ENABLED)
        message(STATUS "command line application ${app_name} -- enabled")
        if (NOT APP_SOURCES)
            set(APP_SOURCES "${app_name}.cpp")
        endif()
        if (TECA_HAS_BOOST)
            list(APPEND APP_SOURCES teca_app_util.cxx)
        endif()
        add_executable(${app_name} ${APP_SOURCES})
        if (APP_LIBS)
            target_link_libraries(${app_name}
                teca_system teca_core teca_data teca_io teca_alg
                ${APP_LIBS})
        endif()
        set_target_properties(${app_name} PROPERTIES APP_TYPE C++)
        install(TARGETS ${app_name} RUNTIME DESTINATION ${BIN_PREFIX})
    else()
        message(STATUS "command line application ${app_name} -- disabled")
    endif()
endfunction()
