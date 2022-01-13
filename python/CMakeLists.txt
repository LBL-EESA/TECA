project(teca_py)

if (TECA_HAS_PYTHON)
    include(CMake/teca_python.cmake)

    depend_swig(teca_py.i teca_py.dep)
    wrap_swig(teca_py.i teca_py.cxx DEPEND_FILES teca_py.dep)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/../HAMR
            ${CMAKE_CURRENT_BINARY_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/../core
            ${CMAKE_CURRENT_SOURCE_DIR}/../data
            ${CMAKE_CURRENT_SOURCE_DIR}/../io
            ${CMAKE_CURRENT_SOURCE_DIR}/../alg
            ${CMAKE_CURRENT_SOURCE_DIR}/../system
            ${CMAKE_CURRENT_SOURCE_DIR}/../HAMR
            )

    PYTHON_ADD_MODULE(_teca_py ${CMAKE_CURRENT_BINARY_DIR}/teca_py.cxx)

    if (TECA_HAS_CUDA)
        set_source_files_properties(teca_py.cxx PROPERTIES LANGUAGE CUDA)
    endif()

    # include the directory containing _teca_py in the rpath
    # this enables one not to have to set (DY)LD_LIBRARY_PATH
    # prior to importing teca.
    if (APPLE)
        set_target_properties(_teca_py PROPERTIES INSTALL_RPATH "@loader_path/./")
    elseif(UNIX)
        set_target_properties(_teca_py PROPERTIES INSTALL_RPATH "\$ORIGIN/./")
    endif()

    target_link_libraries(_teca_py PYTHON
        teca_core teca_data teca_alg teca_io teca_system)

    add_custom_command(TARGET _teca_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/teca_py.py
            ${CMAKE_CURRENT_BINARY_DIR}/../${LIB_PREFIX})

    configure_file(teca.py ${CMAKE_CURRENT_BINARY_DIR}/../${LIB_PREFIX}/teca.py
        COPYONLY)

    install(TARGETS _teca_py LIBRARY DESTINATION ${LIB_PREFIX})
    install(FILES teca.py ${CMAKE_CURRENT_BINARY_DIR}/teca_py.py
        DESTINATION ${LIB_PREFIX})
endif()
