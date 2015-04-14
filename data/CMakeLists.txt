project(teca_data)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(teca_data_srcs
    teca_array_collection.cxx
    teca_cartesian_mesh.cxx
    teca_mesh.cxx
    teca_table.cxx
    teca_uniform_cartesian_mesh.cxx
    )

add_library(teca_data ${teca_data_srcs})
target_link_libraries(teca_data teca_core)

target_include_directories(teca_data
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )

if (BUILD_TESTING)
    add_subdirectory(test)
endif()
