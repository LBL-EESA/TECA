project(teca_core)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(teca_core_srcs
    teca_algorithm.cxx
    teca_algorithm_executive.cxx
    teca_bad_cast.cxx
    teca_binary_stream.cxx
    teca_common.cxx
    teca_dataset.cxx
    teca_dataset_capture.cxx
    teca_index_executive.cxx
    teca_index_reduce.cxx
    teca_memory_profiler.cxx
    teca_metadata.cxx
    teca_metadata_util.cxx
    teca_mpi_manager.cxx
    teca_mpi_util.cxx
    teca_parallel_id.cxx
    teca_profiler.cxx
    teca_programmable_algorithm.cxx
    teca_programmable_reduce.cxx
    teca_string_util.cxx
    teca_system_util.cxx
    teca_threaded_algorithm.cxx
    teca_threaded_programmable_algorithm.cxx
    teca_thread_util.cxx
    )

set(teca_core_link hamr)

set(teca_core_cuda_srcs)
if (TECA_HAS_CUDA)
    set(teca_core_cuda_srcs
        teca_cuda_util.cu
        )
    set_source_files_properties(${teca_core_srcs} PROPERTIES LANGUAGE CUDA)
endif()

if (TECA_HAS_MPI)
    list(APPEND teca_core_link MPI)
endif()

if (TECA_HAS_BOOST)
    include_directories(SYSTEM ${Boost_INCLUDE_DIR})
    list(APPEND teca_core_link ${Boost_LIBRARIES})
endif()

if (TECA_HAS_UDUNITS)
    include_directories(SYSTEM ${UDUNITS_INCLUDE_DIR})
    list(APPEND teca_core_link ${UDUNITS_LIBRARY})
    list(APPEND teca_core_srcs teca_calcalcs.cxx)
endif()

list(APPEND teca_core_link pthread)

add_library(teca_core ${teca_core_srcs} ${teca_core_cuda_srcs})
target_link_libraries(teca_core PUBLIC ${teca_core_link})

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
