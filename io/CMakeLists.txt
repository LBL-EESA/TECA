project(teca_io)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(teca_io_srcs
    teca_file_util.cxx
    teca_table_reader.cxx
    teca_table_writer.cxx
    teca_vtk_util.cxx
    )

if (WIN32)
    list(APPEND teca_io_srcs win_windirent.cxx)
endif()

set(teca_io_link)

if (TECA_HAS_MPI)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
    list(APPEND teca_io_link ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()

if (TECA_HAS_NETCDF)
    list(APPEND teca_io_srcs teca_cf_reader.cxx)
    include_directories(SYSTEM ${NETCDF_INCLUDE_DIRS})
    list(APPEND teca_io_link ${NETCDF_LIBRARIES})
    if (TECA_HAS_OPENSSL)
        include_directories(SYSTEM ${OPENSSL_INCLUDE_DIR})
        list(APPEND teca_io_link ${OPENSSL_LIBRARIES})
    endif()
endif()

if (TECA_HAS_VTK)
    include(${VTK_USE_FILE})
elseif (TECA_HAS_PARAVIEW)
    include(${PARAVIEW_USE_FILE})
endif()
list(APPEND teca_io_srcs teca_vtk_cartesian_mesh_writer.cxx)
if (TECA_HAS_VTK OR TECA_HAS_PARAVIEW)
    list(APPEND teca_io_link ${VTK_LIBRARIES})
endif()

if (TECA_HAS_BOOST)
    include_directories(SYSTEM ${Boost_INCLUDE_DIR})
    list(APPEND teca_io_link ${Boost_LIBRARIES})
endif()

if (TECA_HAS_LIBXLSXWRITER)
    include_directories(SYSTEM ${LIBXLSXWRITER_INCLUDE_DIR})
    list(APPEND teca_io_link ${LIBXLSXWRITER_LIBRARIES})
endif()

add_library(teca_io ${teca_io_srcs})
target_link_libraries(teca_io teca_core teca_data ${teca_io_link})

target_include_directories(teca_io
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_PREFIX}>
    )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${INCLUDE_PREFIX}
    FILES_MATCHING PATTERN "*.h")

install(TARGETS teca_io
    EXPORT teca_io
    INCLUDES DESTINATION ${INCLUDE_PREFIX}
    ARCHIVE DESTINATION ${LIB_PREFIX}
    LIBRARY DESTINATION ${LIB_PREFIX}
    )

install(EXPORT teca_io
    DESTINATION ${LIB_PREFIX}
    FILE teca_io.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    )
