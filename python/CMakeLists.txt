project(teca_py)

if (TECA_HAS_PYTHON)
    include(CMake/teca_python.cmake)

    depend_swig(teca_py.i teca_py.dep)
    wrap_swig(teca_py.i teca_py.cxx teca_py.dep)

    include_directories(SYSTEM ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_DIR})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/../core
            ${CMAKE_CURRENT_SOURCE_DIR}/../data
            ${CMAKE_CURRENT_SOURCE_DIR}/../io
            ${CMAKE_CURRENT_SOURCE_DIR}/../alg
            ${CMAKE_CURRENT_SOURCE_DIR}/../system
            )

    PYTHON_ADD_MODULE(_teca_py ${CMAKE_CURRENT_BINARY_DIR}/teca_py.cxx)

    # include the directory containing _teca_py in the rpath
    # this enables one not to have to set (DY)LD_LIBRARY_PATH
    # prior to importing teca.
    if (APPLE)
        set_target_properties(_teca_py PROPERTIES INSTALL_RPATH "@loader_path/./")
    elseif(UNIX)
        set_target_properties(_teca_py PROPERTIES INSTALL_RPATH "\$ORIGIN/./")
    endif()

    target_link_libraries(_teca_py ${PYTHON_LIBRARIES}
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
