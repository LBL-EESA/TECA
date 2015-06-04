project(teca_io)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(teca_io_srcs
    teca_file_util.cxx
    teca_table_writer.cxx
    )

if (WIN32)
    list(APPEND teca_io_srcs win_windirent.cxx)
endif()

set(teca_io_link)

if (TECA_HAS_MPI)
    include_directories(${MPI_C_INCLUDE_PATH})
    add_definitions(-DTECA_HAS_MPI)
    list(APPEND teca_io_link ${MPI_CXX_LIBRARIES})
endif()

if (TECA_HAS_NETCDF)
    list(APPEND teca_io_srcs teca_cf_reader.cxx)
    include_directories(${NETCDF_INCLUDE_DIRS})
    list(APPEND teca_io_link ${NETCDF_LIBRARIES})
endif()

if (TECA_HAS_VTK)
    include(${VTK_USE_FILE})
    list(APPEND teca_io_srcs teca_vtk_cartesian_mesh_writer.cxx)
    list(APPEND teca_io_link ${VTK_LIBRARIES})
endif()

if (TECA_HAS_BOOST)
    include_directories(${Boost_INCLUDE_PATH})
    add_definitions(-DTECA_HAS_BOOST)
    list(APPEND teca_io_link ${Boost_LIBRARIES})
endif()

add_library(teca_io ${teca_io_srcs})
target_link_libraries(teca_io teca_core teca_data ${teca_io_link})

target_include_directories(teca_io
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )
