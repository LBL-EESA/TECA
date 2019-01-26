project(teca_core)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(teca_core_srcs
    teca_algorithm.cxx
    teca_algorithm_executive.cxx
    teca_binary_stream.cxx
    teca_calendar.cxx
    teca_dataset.cxx
    teca_metadata.cxx
    teca_mpi_manager.cxx
    teca_parallel_id.cxx
    teca_temporal_reduction.cxx
    teca_threaded_algorithm.cxx
    teca_thread_pool.cxx
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

if (TECA_HAS_UDUNITS)
    include_directories(SYSTEM ${UDUNITS_INCLUDE_DIR})
    list(APPEND teca_core_link ${UDUNITS_LIBRARY})
    list(APPEND teca_core_srcs calcalcs.cxx)
endif()

list(APPEND teca_core_link pthread)

add_library(teca_core ${teca_core_srcs})
target_link_libraries(teca_core ${teca_core_link})

target_include_directories(teca_core
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_PREFIX}>
    )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${INCLUDE_PREFIX}
    FILES_MATCHING PATTERN "*.h")

install(TARGETS teca_core
    EXPORT teca_core
    INCLUDES DESTINATION ${INCLUDE_PREFIX}
    ARCHIVE DESTINATION ${LIB_PREFIX}
    LIBRARY DESTINATION ${LIB_PREFIX}
    )

install(EXPORT teca_core
    DESTINATION ${LIB_PREFIX}
    FILE teca_core.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    )
