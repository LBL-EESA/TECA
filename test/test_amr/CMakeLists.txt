project(teca_test_amr)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    )

add_executable(teca_test_amr test_amr_metadata.cpp)
target_link_libraries(teca_test_amr teca_core)

target_include_directories(teca_test_amr
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )
