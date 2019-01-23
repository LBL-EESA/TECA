project(teca_data)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(teca_data_srcs
    teca_array_collection.cxx
    teca_cartesian_mesh.cxx
    teca_coordinate_util.cxx
    teca_mesh.cxx
    teca_table.cxx
    teca_table_collection.cxx
    teca_uniform_cartesian_mesh.cxx
    teca_database.cxx
    )

add_library(teca_data ${teca_data_srcs})
target_link_libraries(teca_data teca_core)

target_include_directories(teca_data
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_PREFIX}>
    )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${INCLUDE_PREFIX}
    FILES_MATCHING PATTERN "*.h")

install(TARGETS teca_data
    EXPORT teca_data
    INCLUDES DESTINATION ${INCLUDE_PREFIX}
    ARCHIVE DESTINATION ${LIB_PREFIX}
    LIBRARY DESTINATION ${LIB_PREFIX}
    )

install(EXPORT teca_data
    DESTINATION ${LIB_PREFIX}
    FILE teca_data.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    )
