function(wrap_swig input output depends)
    if (swig_cmd)
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${output}
            COMMAND ${swig_cmd} -c++ -python -w341
                -I${CMAKE_CURRENT_SOURCE_DIR}/../core
                -I${CMAKE_CURRENT_SOURCE_DIR}/../data
                -I${CMAKE_CURRENT_SOURCE_DIR}/../io
                -I${CMAKE_CURRENT_SOURCE_DIR}/../alg
                -o ${CMAKE_CURRENT_SOURCE_DIR}/${output}
                ${CMAKE_CURRENT_SOURCE_DIR}/${input}
            DEPENDS ${input} ${depends}
            COMMENT "wrapping ${input}...")
    endif()
endfunction()
function(teca_python_module mname deps)
    include_directories(SYSTEM
        ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_DIR})
    wrap_swig(teca_py_${mname}.i teca_py_${mname}.cxx ${deps})
    PYTHON_ADD_MODULE(_teca_py_${mname}
        ${CMAKE_CURRENT_SOURCE_DIR}/teca_py_${mname}.cxx)
    target_link_libraries(_teca_py_${mname} ${PYTHON_LIBRARIES} teca_${mname})
    add_custom_command(TARGET _teca_py_${mname} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/teca_py_${mname}.py
            ${CMAKE_CURRENT_BINARY_DIR}/../lib)
    install(TARGETS _teca_py_${mname} LIBRARY DESTINATION lib)
    install(FILES teca_py_${mname}.py DESTINATION lib)
endfunction()
