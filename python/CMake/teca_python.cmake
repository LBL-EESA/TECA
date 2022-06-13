function(depend_swig input output)
    set(input_file ${CMAKE_CURRENT_SOURCE_DIR}/${input})
    set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${output})
    # custom command to update the dependency file
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
            DYLD_LIBRARY_PATH=$ENV{DYLD_LIBRARY_PATH}
            ${swig_cmd} -c++ -python -MM
            -I${MPI4Py_INCLUDE_DIR}
            -I${CMAKE_CURRENT_BINARY_DIR}
            -I${CMAKE_CURRENT_BINARY_DIR}/../HAMR
            -I${CMAKE_CURRENT_BINARY_DIR}/..
            -I${CMAKE_CURRENT_SOURCE_DIR}/../core
            -I${CMAKE_CURRENT_SOURCE_DIR}/../data
            -I${CMAKE_CURRENT_SOURCE_DIR}/../io
            -I${CMAKE_CURRENT_SOURCE_DIR}/../alg
            -I${CMAKE_CURRENT_SOURCE_DIR}/../system
            -I${CMAKE_CURRENT_SOURCE_DIR}/../HAMR
            ${input_file} | sed -e 's/[[:space:]\\]\\{1,\\}//g' -e '1,2d' -e '/teca_config\\.h/d' > ${output_file}
        MAIN_DEPENDENCY ${input_file}
        COMMENT "Generating dependency file for ${input}...")
    # bootstrap the dependency list
    message(STATUS "Generating initial dependency list for ${input}")
    execute_process(
        COMMAND env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
            DYLD_LIBRARY_PATH=$ENV{DYLD_LIBRARY_PATH}
            ${swig_cmd} -c++ -python -MM
            -I${MPI4Py_INCLUDE_DIR}
            -I${CMAKE_CURRENT_BINARY_DIR}
            -I${CMAKE_CURRENT_BINARY_DIR}/../HAMR
            -I${CMAKE_CURRENT_BINARY_DIR}/..
            -I${CMAKE_CURRENT_SOURCE_DIR}/../core
            -I${CMAKE_CURRENT_SOURCE_DIR}/../data
            -I${CMAKE_CURRENT_SOURCE_DIR}/../io
            -I${CMAKE_CURRENT_SOURCE_DIR}/../alg
            -I${CMAKE_CURRENT_SOURCE_DIR}/../system
            -I${CMAKE_CURRENT_SOURCE_DIR}/../HAMR
            ${input_file}
       COMMAND sed -e s/[[:space:]\\]\\{1,\\}//g -e 1,2d -e /teca_config\\.h/d
       OUTPUT_VARIABLE cpp_deps)
    # include the files brought into the module via pythoncode directive
    # swig -MM doesn't pick these up. below we grep the .i files for them
    message(STATUS "Generating pythoncode dependency list for ${input}...")
    set(all_deps ${cpp_deps})
    set(comps core data io alg system)
    foreach(comp ${comps})
        set(i_file ${CMAKE_CURRENT_SOURCE_DIR}/teca_py_${comp}.i)
        set(comp_path ${CMAKE_CURRENT_SOURCE_DIR}/../${comp})
        string(REPLACE "/" "\\/" comp_path ${comp_path})
        set(pycode_deps)
        execute_process(
            COMMAND grep -rI "pythoncode \"teca_.*\\.py\"\$" ${i_file}
            COMMAND sed "s/.*pythoncode \"\\\(teca_.*\\.py\\\)\"$/${comp_path}\\/\\1/g"
            OUTPUT_VARIABLE pycode_deps)
        string(APPEND all_deps ${pycode_deps})
        add_custom_command(
            OUTPUT ${output_file}
            COMMAND grep -rI "'pythoncode \"teca_.*\\.py\"\$'" ${i_file} | sed
                "'s/.*pythoncode \"\\\(teca_.*\\.py\\\)\"$/${comp_path}\\/\\1/g'" >> ${output_file}
            MAIN_DEPENDENCY ${i_file}
            COMMENT "Generating dependency file for ${i_file}...")
    endforeach()
    file(WRITE ${output_file} ${all_deps})
endfunction()
function(wrap_swig input output)
    set(opt_args)
    set(val_args)
    set(array_args DEPEND_FILES)
    cmake_parse_arguments(wrap_swig
        "${opt_args}" "${val_args}" "${array_args}" ${ARGN})
    set(input_file ${CMAKE_CURRENT_SOURCE_DIR}/${input})
    set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${output})
    set(depends)
    set(depend_files)
    foreach (depend_file ${wrap_swig_DEPEND_FILES})
        set(tmp)
        file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/${depend_file} tmp)
        list(APPEND depends ${tmp})
        list(APPEND depend_files ${CMAKE_CURRENT_BINARY_DIR}/${depend_file})
    endforeach()
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
            DYLD_LIBRARY_PATH=$ENV{DYLD_LIBRARY_PATH}
            ${swig_cmd} -c++ -python -threads -w341,325
            -DSWIG_TYPE_TABLE=teca_py
            -I${MPI4Py_INCLUDE_DIR}
            -I${CMAKE_CURRENT_BINARY_DIR}
            -I${CMAKE_CURRENT_BINARY_DIR}/../HAMR
            -I${CMAKE_CURRENT_BINARY_DIR}/..
            -I${CMAKE_CURRENT_SOURCE_DIR}/../core
            -I${CMAKE_CURRENT_SOURCE_DIR}/../data
            -I${CMAKE_CURRENT_SOURCE_DIR}/../io
            -I${CMAKE_CURRENT_SOURCE_DIR}/../alg
            -I${CMAKE_CURRENT_SOURCE_DIR}/../system
            -I${CMAKE_CURRENT_SOURCE_DIR}/../HAMR
            -o ${output_file} ${input_file}
        MAIN_DEPENDENCY ${input_file}
        DEPENDS ${depend_files} ${depends}
        COMMENT "Generating python bindings for ${input}...")
endfunction()
