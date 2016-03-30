cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0022 NEW)

project(teca_core)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(teca_core_srcs
    teca_algorithm.cxx
    teca_algorithm_executive.cxx
    teca_binary_stream.cxx
    teca_calendar.cxx
    teca_metadata.cxx
    teca_parallel_id.cxx
    teca_temporal_reduction.cxx
    teca_threaded_algorithm.cxx
    teca_time_step_executive.cxx
    teca_variant_array.cxx
    )

set(teca_core_link)

if (TECA_HAS_MPI)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
    list(APPEND teca_core_link ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()

if (TECA_HAS_BOOST)
    include_directories(SYSTEM ${Boost_INCLUDE_DIR})
    list(APPEND teca_core_link ${Boost_LIBRARIES})
endif()

add_library(teca_core ${teca_core_srcs})

target_link_libraries(teca_core ${teca_core_link})

target_include_directories(teca_core
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION include
    FILES_MATCHING PATTERN "*.h")

install(TARGETS teca_core
    EXPORT teca_core
    INCLUDES DESTINATION include
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    )

install(EXPORT teca_core
    DESTINATION lib
    FILE teca_core.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    )
