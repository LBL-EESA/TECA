project(teca_apps)
cmake_policy(SET CMP0022 NEW)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_alg,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(teca_app_link)

if (TECA_HAS_MPI)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
    list(APPEND teca_app_link ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()

if (TECA_HAS_BOOST AND TECA_HAS_NETCDF)
    include_directories(SYSTEM ${Boost_INCLUDE_DIR})
    list(APPEND teca_app_link ${Boost_LIBRARIES})

    add_executable(teca_ar_detect teca_ar_detect.cpp)
    target_link_libraries(teca_ar_detect
        teca_core teca_data teca_io teca_alg
        ${teca_app_link}
        )
    install(TARGETS teca_ar_detect
        RUNTIME DESTINATION bin
        )

    add_executable(teca_tc_detect teca_tc_detect.cpp)
    target_link_libraries(teca_tc_detect
        teca_core teca_data teca_io teca_alg
        ${teca_app_link}
        )
    install(TARGETS teca_tc_detect
        RUNTIME DESTINATION bin
        )
endif()

