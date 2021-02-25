function(teca_py_install)
    if (TECA_HAS_PYTHON)
        set(pysrcs ${ARGV})
        list(GET pysrcs 0 dest)
        list(REMOVE_AT pysrcs 0)
        # pass the list of python based applications
        # this function copies the files into the build binary dir,
        # sets up the install, and py-compiles the sources
        foreach(pysrc ${pysrcs})
            configure_file(${pysrc}
                ${CMAKE_CURRENT_BINARY_DIR}/../${dest}/${pysrc}
                COPYONLY)
        endforeach()
        install(FILES ${pysrcs} DESTINATION ${dest})
        # TODO compile the sources
    endif()
endfunction()

function(teca_py_install_apps)
    if (TECA_HAS_PYTHON)
        # pass the list of python based applications
        # this function copies the files into the build binary dir,
        # sets up the install, and py-compiles the sources
        set(pysrcs ${ARGV})
        set(pyapps)
        foreach(pysrc ${pysrcs})
            get_filename_component(appname ${pysrc} NAME_WE)
            configure_file(${pysrc}
                ${CMAKE_CURRENT_BINARY_DIR}/../${BIN_PREFIX}/${appname}
                @ONLY)
            list(APPEND pyapps
                ${CMAKE_CURRENT_BINARY_DIR}/../${BIN_PREFIX}/${appname})
        endforeach()
        install(PROGRAMS ${pyapps} DESTINATION ${BIN_PREFIX})
        # TODO compile the sources
    endif()
endfunction()

# teca_add_python_app(name
#   SOURCES  -- optional, source files to comile
#   FEATURES -- optional, boolean condition decribing feature dependencies
#   )
function (teca_add_python_app app_name)
    if (TECA_HAS_PYTHON)
        set(opt_args)
        set(val_args)
        set(array_args SOURCES FEATURES)
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
                set(APP_SOURCES "${app_name}.in")
            endif()
            add_custom_target(${app_name})
            set_target_properties(${app_name} PROPERTIES APP_TYPE Python)
            teca_py_install_apps(${APP_SOURCES})
        else()
            message(STATUS "command line application ${app_name} -- disabled")
        endif()
    endif()
endfunction()
