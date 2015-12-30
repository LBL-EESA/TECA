function(depend_swig input output)
    set(input_file ${CMAKE_CURRENT_SOURCE_DIR}/${input})
    set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${output})
    # custom command to update the dependency file
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${swig_cmd} -c++ -python -MM
            -I${CMAKE_CURRENT_BINARY_DIR}
            -I${CMAKE_CURRENT_BINARY_DIR}/..
            -I${CMAKE_CURRENT_SOURCE_DIR}/../core
            -I${CMAKE_CURRENT_SOURCE_DIR}/../data
            -I${CMAKE_CURRENT_SOURCE_DIR}/../io
            -I${CMAKE_CURRENT_SOURCE_DIR}/../alg
            ${input_file} | sed -e 's/[[:space:]\\]\\{1,\\}//g' -e '1,2d' -e '/teca_config\\.h/d' > ${output_file}
        MAIN_DEPENDENCY ${input_file}
        COMMENT "Generating dependency file for ${input}...")
    # bootstrap the dependency list
    message(STATUS "Generating initial dependency list for ${input}")
    execute_process(
        COMMAND ${swig_cmd} -c++ -python -MM
            -I${CMAKE_CURRENT_BINARY_DIR}
            -I${CMAKE_CURRENT_BINARY_DIR}/..
            -I${CMAKE_CURRENT_SOURCE_DIR}/../core
            -I${CMAKE_CURRENT_SOURCE_DIR}/../data
            -I${CMAKE_CURRENT_SOURCE_DIR}/../io
            -I${CMAKE_CURRENT_SOURCE_DIR}/../alg
            ${input_file}
       COMMAND sed -e s/[[:space:]\\]\\{1,\\}//g -e 1,2d -e /teca_config\\.h/d
       OUTPUT_FILE ${output_file})
endfunction()
function(wrap_swig input output depend)
    set(input_file ${CMAKE_CURRENT_SOURCE_DIR}/${input})
    set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${output})
    set(depend_file ${CMAKE_CURRENT_BINARY_DIR}/${depend})
    file(STRINGS ${depend_file} depends)
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${swig_cmd} -c++ -python -w341,325
            -DSWIG_TYPE_TABLE=teca_py
            -I${CMAKE_CURRENT_BINARY_DIR}
            -I${CMAKE_CURRENT_BINARY_DIR}/..
            -I${CMAKE_CURRENT_SOURCE_DIR}/../core
            -I${CMAKE_CURRENT_SOURCE_DIR}/../data
            -I${CMAKE_CURRENT_SOURCE_DIR}/../io
            -I${CMAKE_CURRENT_SOURCE_DIR}/../alg
            -o ${output_file} ${input_file}
        MAIN_DEPENDENCY ${input_file}
        DEPENDS ${depend_file} ${depends}
        COMMENT "Generating python bindings for ${input}...")
endfunction()
function(teca_python_module mname)
    depend_swig(teca_py_${mname}.i teca_py_${mname}.dep)
    wrap_swig(teca_py_${mname}.i teca_py_${mname}.cxx teca_py_${mname}.dep)
    include_directories(SYSTEM ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_DIR})
    PYTHON_ADD_MODULE(_teca_py_${mname} ${CMAKE_CURRENT_BINARY_DIR}/teca_py_${mname}.cxx)
    target_link_libraries(_teca_py_${mname} ${PYTHON_LIBRARIES} teca_${mname})
    add_custom_command(TARGET _teca_py_${mname} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/teca_py_${mname}.py
            ${CMAKE_CURRENT_BINARY_DIR}/../lib)
    install(TARGETS _teca_py_${mname} LIBRARY DESTINATION lib)
    install(FILES teca_py_${mname}.py DESTINATION lib)
endfunction()
