cmake_minimum_required(VERSION 2.8)
project(teca_alg)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(teca_alg_srcs
    teca_ar_detect.cxx
    )

add_library(teca_alg ${teca_alg_srcs})
target_link_libraries(teca_alg teca_data teca_core)

target_include_directories(teca_alg
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )

#if (BUILD_TESTING)
#    add_subdirectory(test)
#endif()
