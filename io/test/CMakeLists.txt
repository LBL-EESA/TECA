project(teca_io_test)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    )

if (NETCDF_FOUND AND VTK_FOUND)
    add_executable(
        test_cf_reader
        test_cf_reader.cpp
        )
    target_link_libraries(test_cf_reader teca_core teca_data teca_io)
endif()
