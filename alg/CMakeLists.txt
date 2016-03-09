cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0022 NEW)

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
    teca_connected_components.cxx
    teca_derived_quantity.cxx
    teca_descriptive_statistics.cxx
    teca_l2_norm.cxx
    teca_mask.cxx
    teca_programmable_algorithm.cxx
    teca_table_reduce.cxx
    teca_table_sort.cxx
    teca_table_to_stream.cxx
    teca_temporal_average.cxx
    teca_vorticity.cxx
    teca_dataset_diff.cxx
    )

set(teca_alg_link pthread)

if (TECA_HAS_MPI)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
    list(APPEND teca_alg_link ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()

if (TECA_HAS_BOOST)
    include_directories(SYSTEM ${Boost_INCLUDE_DIR})
    list(APPEND teca_alg_link ${Boost_LIBRARIES})
endif()

add_library(teca_alg ${teca_alg_srcs})
target_link_libraries(teca_alg teca_io teca_data teca_core ${teca_alg_link})

target_include_directories(teca_alg
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION include
    FILES_MATCHING PATTERN "*.h")

install(TARGETS teca_alg
    EXPORT teca_alg
    INCLUDES DESTINATION include
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    )

install(EXPORT teca_alg
    DESTINATION lib
    FILE teca_alg.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    )

if (TECA_HAS_PYTHON)
    teca_python_module(alg)
endif()
