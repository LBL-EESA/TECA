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
            -I${CMAKE_CURRENT_SOURCE_DIR}/../system
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
            -I${CMAKE_CURRENT_SOURCE_DIR}/../system
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
        COMMAND ${swig_cmd} -c++ -python -threads -w341,325
            -DSWIG_TYPE_TABLE=teca_py
            -I${CMAKE_CURRENT_BINARY_DIR}
            -I${CMAKE_CURRENT_BINARY_DIR}/..
            -I${CMAKE_CURRENT_SOURCE_DIR}/../core
            -I${CMAKE_CURRENT_SOURCE_DIR}/../data
            -I${CMAKE_CURRENT_SOURCE_DIR}/../io
            -I${CMAKE_CURRENT_SOURCE_DIR}/../alg
            -I${CMAKE_CURRENT_SOURCE_DIR}/../system
            -o ${output_file} ${input_file}
        MAIN_DEPENDENCY ${input_file}
        DEPENDS ${depend_file} ${depends}
        COMMENT "Generating python bindings for ${input}...")
endfunction()
