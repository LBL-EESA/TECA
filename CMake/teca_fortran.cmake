function(teca_fortran_preprocess f_type c_types generic_srcs output_var)
    set(output)
    foreach(generic_src ${generic_srcs})
        foreach(c_type ${c_types})
            set(iso_c_type "${f_type}(c_${c_type})")
            configure_file(${generic_src}.f90.in ${generic_src}_${c_type}.f90 @ONLY)
            list(APPEND output ${generic_src}_${c_type}.f90)
        endforeach()
    endforeach()
    set(${output_var} ${${output_var}} ${output} PARENT_SCOPE)
endfunction()


teca_fortran_preprocess(integer "int;long" teca_vector teca_alg_f90_srcs)
teca_fortran_preprocess(real "float;double"
    "teca_vector;teca_distance_func" teca_alg_f90_srcs)
