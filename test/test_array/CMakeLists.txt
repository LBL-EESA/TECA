project(teca_test_array)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(array_test_srcs
    array.cxx
    array_add.cxx
    array_executive.cxx
    array_scalar_multiply.cxx
    array_source.cxx
    array_temporal_stats.cxx
    array_time_average.cxx
    array_writer.cxx
    )

add_library(teca_test_array ${array_test_srcs})
target_link_libraries(teca_test_array teca_core)

target_include_directories(teca_test_array
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )
