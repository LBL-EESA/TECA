cmake_minimum_required(VERSION 2.8)
project(teca_alg)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(teca_alg_srcs
    teca_ar_detect.cxx
    teca_cartesian_mesh_subset.cxx
    teca_cartesian_mesh_regrid.cxx
    teca_programmable_source.cxx
    teca_table_reduce.cxx
    teca_temporal_average.cxx
    )

if (TECA_HAS_MPI)
    include_directories(${MPI_C_INCLUDE_PATH})
    add_definitions(-DTECA_HAS_MPI)
endif()

add_library(teca_alg ${teca_alg_srcs})
target_link_libraries(teca_alg teca_io teca_data teca_core)
if (TECA_HAS_MPI)
    target_link_libraries(teca_alg ${MPI_CXX_LIBRARIES})
endif()

target_include_directories(teca_alg
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )
