add_library(teca_timer STATIC teca_memory_profiler.cxx teca_timer.cxx)

target_include_directories(teca_timer SYSTEM INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/teca_timer>)

set(teca_timer_link)

if (TECA_HAS_MPI)
    list(APPEND teca_timer_link MPI)
endif()

if (TECA_HAS_VTK)
    list(APPEND teca_timer_link MPI)
endif()

target_link_libraries(teca_timer PUBLIC ${teca_timer_link})

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION include/teca_timer FILES_MATCHING PATTERN "*.h")

install(TARGETS teca_timer EXPORT teca_timer
  INCLUDES DESTINATION include/teca_timer ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(EXPORT teca_timer DESTINATION lib/cmake
  EXPORT_LINK_INTERFACE_LIBRARIES)
