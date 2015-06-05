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
    add_definitions(-DTECA_HAS_MPI)
    include_directories(${MPI_C_INCLUDE_PATH})
    list(APPEND teca_io_link ${MPI_CXX_LIBRARIES})
endif()

if (TECA_HAS_BOOST)
    add_definitions(-DTECA_HAS_BOOST)
    include_directories(${Boost_INCLUDE_DIR})
    list(APPEND teca_app_link ${Boost_LIBRARIES})
endif()

if (TECA_HAS_NETCDF)
    add_executable(teca_ar_detect teca_ar_detect.cpp)
    target_link_libraries(teca_ar_detect
        teca_core teca_data teca_io teca_alg
        ${teca_app_link}
        )
    install(TARGETS teca_ar_detect
        RUNTIME DESTINATION bin
        )
endif()

