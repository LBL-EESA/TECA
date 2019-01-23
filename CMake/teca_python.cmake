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
                COPYONLY)
            list(APPEND pyapps
                ${CMAKE_CURRENT_BINARY_DIR}/../${BIN_PREFIX}/${appname})
        endforeach()
        install(PROGRAMS ${pyapps} DESTINATION ${BIN_PREFIX})
        # TODO compile the sources
    endif()
endfunction()
