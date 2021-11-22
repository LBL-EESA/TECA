project(teca_test_array)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(array_test_cxx_srcs
    array.cxx
    array_add.cxx
    array_add_internals.cxx
    array_executive.cxx
    array_scalar_multiply.cxx
    array_scalar_multiply_internals.cxx
    array_source.cxx
    array_source_internals.cxx
    array_temporal_stats.cxx
    array_temporal_stats_internals.cxx
    array_time_average.cxx
    array_writer.cxx
    )

if (TECA_HAS_MPI)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
endif()

set(array_test_cuda_srcs)
if (TECA_HAS_CUDA)
    set(array_test_cuda_srcs
        array_add_internals.cu
        array_temporal_stats_internals.cu
        array_scalar_multiply_internals.cu
        array_source_internals.cu
        )
    set_source_files_properties(${array_test_cxx_srcs} PROPERTIES LANGUAGE CUDA)
endif()

add_library(teca_test_array ${array_test_cxx_srcs} ${array_test_cuda_srcs})
target_link_libraries(teca_test_array teca_core hamr)

target_include_directories(teca_test_array
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )
