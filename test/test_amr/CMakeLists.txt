project(teca_test_amr)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    )

if (TECA_HAS_MPI)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
endif()

if (TECA_HAS_CUDA)
    set_source_files_properties(test_amr_metadata.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(teca_test_amr test_amr_metadata.cpp)
target_link_libraries(teca_test_amr teca_core)

target_include_directories(teca_test_amr
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )
